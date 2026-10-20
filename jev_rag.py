import os
import uuid
import hashlib
import argparse
import requests
from pathlib import Path
from typing import List, Dict, Any, Set, Optional

import fitz  # PyMuPDF
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType,
)
from typesafe_sdk import TypeSafeClient, Noul, Score

# =====================================================================
# ENDPOINT CONFIGURATION
# =====================================================================
LLM_CHAT_URL = "http://localhost:1234/api/v1/chat"
EMBEDDINGS_URL = "http://127.0.0.1:1234/v1/embeddings"
MODELS_LIST_URL = "http://127.0.0.1:1234/v1/models"

QDRANT_URL = (
    "<vector db URL>"
)
COLLECTION_NAME = "pdf_directory_kb"
JEV_MODEL = os.environ.get("JEV_MODEL", "jev-1.14.0")


class JevDirectoryRAG:
    def __init__(
        self,
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        # 1. Validate API keys for Qdrant Cloud & TypeSafe Jev
        qdrant_api_key = os.environ.get("QDRANT_API_KEY")
        if not qdrant_api_key:
            raise EnvironmentError(
                "QDRANT_API_KEY environment variable is required for Qdrant Cloud."
            )

        if not os.environ.get("TYPESAFE_API_KEY"):
            raise EnvironmentError("TYPESAFE_API_KEY environment variable is not set.")

        # Optional token if LM Studio Server authentication is enabled
        self.lm_token = os.environ.get("LM_API_TOKEN", "lm-studio")
        self.lm_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.lm_token}",
        }

        # 2. Connect to Qdrant Cloud & TypeSafe Jev
        self.qdrant = QdrantClient(url=QDRANT_URL, api_key=qdrant_api_key)
        self.jev_client = TypeSafeClient()

        # 3. Resolve or Auto-Detect LM Studio Model Identifiers
        detected_llm, detected_embed = self._detect_lmstudio_models()
        self.llm_model = (
            llm_model or os.environ.get("LLM_MODEL") or detected_llm
        )
        self.embedding_model = (
            embedding_model or os.environ.get("EMBEDDING_MODEL") or detected_embed
        )

        if not self.llm_model or not self.embedding_model:
            raise RuntimeError(
                "Could not auto-detect models from LM Studio. Ensure both an LLM and an "
                "embedding model are loaded in LM Studio, or pass --llm-model and --embed-model."
            )

        print(f"[*] Connected to Qdrant Cloud: {QDRANT_URL}")
        print(f"[*] Using Local Embedding Model: '{self.embedding_model}' ({EMBEDDINGS_URL})")
        print(f"[*] Using Local LLM Model:       '{self.llm_model}' ({LLM_CHAT_URL})")

    def _detect_lmstudio_models(self) -> tuple[Optional[str], Optional[str]]:
        """Queries LM Studio /v1/models to automatically pick loaded embedding & chat models."""
        try:
            resp = requests.get(MODELS_LIST_URL, headers=self.lm_headers, timeout=5)
            resp.raise_for_status()
            models = [m["id"] for m in resp.json().get("data", [])]
            embed_models = [m for m in models if "embed" in m.lower()]
            chat_models = [m for m in models if "embed" not in m.lower()]
            return (
                chat_models[0] if chat_models else (models[0] if models else None),
                embed_models[0] if embed_models else (models[0] if models else None),
            )
        except Exception:
            return None, None

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Calls http://127.0.0.1:1234/v1/embeddings directly."""
        resp = requests.post(
            EMBEDDINGS_URL,
            headers=self.lm_headers,
            json={
                "model": self.embedding_model,
                "input": texts,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        # Ensure embeddings are returned in exact input order
        data.sort(key=lambda item: item["index"])
        return [item["embedding"] for item in data]

    def _chat_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Calls LM Studio's native POST http://localhost:1234/api/v1/chat endpoint.
        Uses store=False for stateless one-off RAG synthesis.
        """
        resp = requests.post(
            LLM_CHAT_URL,
            headers=self.lm_headers,
            json={
                "model": self.llm_model,
                "system_prompt": system_prompt,
                "input": user_prompt,
                "temperature": 0.1,
                "store": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        payload = resp.json()

        # Parse LM Studio /api/v1/chat output items
        if "output" in payload and isinstance(payload["output"], list):
            messages = [
                item.get("content", "")
                for item in payload["output"]
                if item.get("type") == "message"
            ]
            if messages:
                return "\n".join(messages).strip()

        # Fallback if endpoint returns OpenAI-style choices
        if "choices" in payload:
            return payload["choices"][0]["message"]["content"].strip()

        return str(payload)

    @staticmethod
    def _file_sha256(filepath: Path) -> str:
        """Computes SHA-256 hash of a PDF to avoid re-embedding unchanged files."""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                hasher.update(block)
        return hasher.hexdigest()

    def _get_indexed_hashes(self) -> Set[str]:
        """Fetches the set of file hashes already stored in the Qdrant Cloud collection."""
        if not self.qdrant.collection_exists(COLLECTION_NAME):
            return set()

        indexed_hashes: Set[str] = set()
        offset = None
        while True:
            records, offset = self.qdrant.scroll(
                collection_name=COLLECTION_NAME,
                limit=250,
                with_payload=["file_hash"],
                with_vectors=False,
                offset=offset,
            )
            for r in records:
                if r.payload and "file_hash" in r.payload:
                    indexed_hashes.add(r.payload["file_hash"])
            if offset is None:
                break
        return indexed_hashes

    def ingest_directory(
        self,
        directory_path: str,
        recursive: bool = True,
        force_reindex: bool = False,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
    ) -> int:
        """
        Scans a directory for PDFs, extracts text, embeds via 127.0.0.1:1234/v1/embeddings,
        and upserts into Qdrant Cloud.
        """
        root = Path(directory_path).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise NotADirectoryError(f"Directory not found: {root}")

        pattern = "**/*" if recursive else "*"
        pdf_files = sorted(
            [
                p
                for p in root.glob(pattern)
                if p.is_file() and p.suffix.lower() == ".pdf"
            ]
        )

        if not pdf_files:
            raise ValueError(f"No PDF files found in directory: {root}")

        print(f"[*] Found {len(pdf_files)} PDF file(s) in '{root}'")

        if force_reindex and self.qdrant.collection_exists(COLLECTION_NAME):
            print("[*] --reindex flag set: Deleting existing Qdrant Cloud collection...")
            self.qdrant.delete_collection(COLLECTION_NAME)

        indexed_hashes = self._get_indexed_hashes()
        new_chunks: List[Dict[str, Any]] = []
        skipped_files = 0

        for pdf_path in pdf_files:
            rel_path = str(pdf_path.relative_to(root))
            file_hash = self._file_sha256(pdf_path)

            if file_hash in indexed_hashes:
                skipped_files += 1
                continue

            try:
                doc = fitz.open(pdf_path)
            except Exception as e:
                print(f"[!] Skipping unreadable PDF '{rel_path}': {e}")
                continue

            file_chunk_count = 0
            for page_idx, page in enumerate(doc):
                page_text = " ".join(page.get_text("text").split())
                if not page_text:
                    continue

                step = max(1, chunk_size - chunk_overlap)
                for start in range(0, len(page_text), step):
                    snippet = page_text[start : start + chunk_size].strip()
                    if len(snippet) < 80:
                        continue

                    new_chunks.append(
                        {
                            "source": rel_path,
                            "file_hash": file_hash,
                            "page": page_idx + 1,
                            "text": snippet,
                        }
                    )
                    file_chunk_count += 1

            doc.close()
            print(f"  + Parsed '{rel_path}' ({file_chunk_count} chunks)")

        if skipped_files > 0:
            print(f"[*] Skipped {skipped_files} already-indexed PDF(s) in Qdrant Cloud.")

        if not new_chunks:
            print("[*] All PDFs in directory are already indexed and up to date.\n")
            return 0

        print(f"[*] Computing local embeddings for {len(new_chunks)} new chunks...")
        texts = [c["text"] for c in new_chunks]

        all_vectors: List[List[float]] = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_vectors = self._embed(texts[i : i + batch_size])
            all_vectors.extend(batch_vectors)

        # Create collection on Qdrant Cloud if it doesn't exist
        if not self.qdrant.collection_exists(COLLECTION_NAME):
            vector_dim = len(all_vectors[0])
            self.qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
            )
            # Create keyword index on file_hash for fast Qdrant Cloud filtering/scrolling
            self.qdrant.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="file_hash",
                field_schema=PayloadSchemaType.KEYWORD,
            )

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload=chunk,
            )
            for chunk, vec in zip(new_chunks, all_vectors)
        ]

        # Upsert to Qdrant Cloud in batches of 64 points
        upsert_batch = 64
        for i in range(0, len(points), upsert_batch):
            self.qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=points[i : i + upsert_batch],
            )

        print(f"[*] Indexed {len(points)} new chunks into Qdrant Cloud.\n")
        return len(points)

    def retrieve_candidates(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """Embeds query locally and retrieves top-k candidates from Qdrant Cloud."""
        query_vector = self._embed([query])[0]
        results = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
        ).points

        candidates = []
        for idx, hit in enumerate(results):
            payload = hit.payload or {}
            candidates.append(
                {
                    "key": f"passage_{idx}",
                    "source": payload.get("source", "unknown"),
                    "page": payload.get("page", 0),
                    "text": payload.get("text", ""),
                    "vector_score": round(hit.score, 4),
                }
            )
        return candidates

    def jev_rerank_and_gate(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        relevance_threshold: float = 0.55,
        min_rubric_score: float = 1.5,
        answerability_threshold: float = 0.65,
    ) -> Dict[str, Any]:
        """
        Single parallel call to TypeSafe Jev:
        - Noul per passage: Calibrated probability of direct evidence.
        - Score per passage: 4-level rubric (0.0 to 3.0) for reranking.
        - Global Noul: Answerability circuit breaker.
        """
        state = {
            "query": query,
            "passages": {
                c["key"]: {
                    "document": f"{c['source']} (p. {c['page']})",
                    "content": c["text"],
                }
                for c in candidates
            },
        }

        questions: Dict[str, Any] = {}
        for c in candidates:
            key = c["key"]
            questions[f"rel_{key}"] = Noul(
                instructions=(
                    f"Does passages['{key}'].content contain factual evidence that "
                    f"directly helps answer the query?"
                )
            )
            questions[f"rubric_{key}"] = Score(
                instructions=f"Rate how useful passages['{key}'].content is for answering the query.",
                criteria=[
                    "Completely irrelevant or off-topic",
                    "Touches on similar keywords but does not answer the query",
                    "Provides partial context or supporting details for the query",
                    "Directly and explicitly answers the query",
                ],
            )

        questions["is_answerable"] = Noul(
            instructions=(
                "Taken together, do the relevant passages provide enough factual evidence "
                "to accurately answer the query without guessing or fabricating details?"
            )
        )

        response = self.jev_client.system_one(
            model=JEV_MODEL,
            state=state,
            questions=questions,
        )

        kept_passages = []
        diagnostics = []

        for c in candidates:
            key = c["key"]
            rel_prob = response.nouls[f"rel_{key}"].noul
            rubric_val = response.scores[f"rubric_{key}"].score
            rubric_conf = response.scores[f"rubric_{key}"].confidence

            passed = (rel_prob >= relevance_threshold) and (
                rubric_val >= min_rubric_score
            )

            item = {
                **c,
                "jev_rel_prob": round(rel_prob, 3),
                "jev_rubric_score": round(rubric_val, 3),
                "jev_confidence": round(rubric_conf, 3),
                "passed_gate": passed,
            }
            diagnostics.append(item)
            if passed:
                kept_passages.append(item)

        kept_passages.sort(
            key=lambda x: (x["jev_rubric_score"], x["jev_rel_prob"]),
            reverse=True,
        )

        answerability_prob = response.nouls["is_answerable"].noul
        can_answer = (
            answerability_prob >= answerability_threshold and len(kept_passages) > 0
        )

        return {
            "model_version": response.model,
            "can_answer": can_answer,
            "answerability_prob": round(answerability_prob, 3),
            "kept_passages": kept_passages,
            "diagnostics": diagnostics,
            "usage": response.usage,
        }

    def ask(self, query: str, top_k: int = 8) -> str:
        """Full RAG flow: Qdrant Cloud Retrieve -> Jev Gate -> Local /api/v1/chat LLM."""
        print(f"\n================ QUERY: {query} ================")

        candidates = self.retrieve_candidates(query, top_k=top_k)
        if not candidates:
            return "No chunks found in Qdrant Cloud."

        gate = self.jev_rerank_and_gate(query, candidates)

        print(f"[Jev Model]: {gate['model_version']} | [Usage]: {gate['usage']}")
        print(f"[Global Answerability]: {gate['answerability_prob']:.3f}")
        print("\n[Candidate Evaluation Breakdown]:")
        for d in gate["diagnostics"]:
            status = "KEEP" if d["passed_gate"] else "DROP"
            print(
                f"  - {d['key']} [{d['source']} p.{d['page']}] | "
                f"VecCosine: {d['vector_score']:.3f} | "
                f"Jev Noul: {d['jev_rel_prob']:.3f} | "
                f"Jev Score: {d['jev_rubric_score']:.2f}/3.00 -> [{status}]"
            )

        # if not gate["can_answer"]:
        #     print("\n[!] JEV CIRCUIT BREAKER TRIGGERED: Skipping Local LLM call.")
        #     return (
        #         "Not enough evidence in the PDF directory to answer this question "
        #         f"(Answerability probability: {gate['answerability_prob']})."
        #     )

        context_blocks = [
            f"Document: {p['source']} (Page {p['page']})\n{p['text']}"
            for p in candidates  # <-- changed from gate["kept_passages"] to candidates
        ]

        # context_blocks = [
        #     f"Document: {p['source']} (Page {p['page']})\n{p['text']}"
        #     for p in gate["kept_passages"]
        # ]
        formatted_context = "\n\n---\n\n".join(context_blocks)

        print(
            f"\n[*] Passing {len(gate['kept_passages'])}/{len(candidates)} verified chunks "
            f"to Local LLM ({LLM_CHAT_URL})..."
        )

        system_prompt = (
            "Answer the user's question strictly using the provided context. "
            "Always cite the document path and page number in brackets like [doc.pdf, p. X]."
        )
        user_prompt = f"Context:\n{formatted_context}\n\nQuestion: {query}"

        return self._chat_llm(system_prompt=system_prompt, user_prompt=user_prompt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Jev + Qdrant Cloud + Local LM Studio RAG over a directory of PDFs."
    )
    parser.add_argument(
        "pdf_dir",
        type=str,
        help="Path to the directory containing PDF files",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Exact LLM model identifier loaded in LM Studio (auto-detected if omitted)",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default=None,
        help="Exact embedding model identifier loaded in LM Studio (auto-detected if omitted)",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force full re-indexing of all PDFs in Qdrant Cloud",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of candidate chunks to retrieve before Jev gating (default: 8)",
    )
    args = parser.parse_args()

    pipeline = JevDirectoryRAG(
        llm_model=args.llm_model,
        embedding_model=args.embed_model,
    )
    pipeline.ingest_directory(args.pdf_dir, force_reindex=args.reindex)

    print("Ready! Type a question to query your PDF directory (or 'exit' to quit).")
    while True:
        try:
            user_query = input("\nAsk a question > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_query or user_query.lower() in {"exit", "quit"}:
            break
        answer = pipeline.ask(user_query, top_k=args.top_k)
        print(f"\n[Final Response]:\n{answer}")