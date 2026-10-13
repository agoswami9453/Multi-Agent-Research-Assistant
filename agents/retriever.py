import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import VECTOR_STORE_DIR, UPLOADS_DIR, EMBEDDING_MODEL_NAME, HIGH_CONFIDENCE_THRESHOLD, MEDIUM_CONFIDENCE_THRESHOLD
from database import save_document_metadata, clear_documents_metadata, get_all_document_metadata
from utils import get_logger

logger = get_logger("RetrieverAgent")

class RetrieverAgent:
    """Agent responsible for document parsing, chunking, embedding, FAISS indexing, and semantic search."""

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.vector_store_path = VECTOR_STORE_DIR / "faiss_index"
        self.vector_store: Optional[FAISS] = None
        self._load_existing_index()

    def _load_existing_index(self):
        """Loads FAISS index from disk if present."""
        if (self.vector_store_path / "index.faiss").exists():
            try:
                self.vector_store = FAISS.load_local(
                    folder_path=str(self.vector_store_path),
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("Loaded existing FAISS vector index from disk.")
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")
                self.vector_store = None

    def ingest_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Parses files, splits into chunks, creates/updates FAISS vector store, and saves metadata."""
        all_docs = []
        doc_summary = []

        for file_path in file_paths:
            path = Path(file_path)
            ext = path.suffix.lower()
            docs = []

            try:
                if ext == ".pdf":
                    loader = PyPDFLoader(str(path))
                    docs = loader.load()
                elif ext in [".docx", ".doc"]:
                    # Try docx2txt or fall back to basic text extraction
                    try:
                        import docx2txt
                        text = docx2txt.process(str(path))
                        from langchain.schema import Document
                        docs = [Document(page_content=text, metadata={"source": path.name, "page": 1})]
                    except Exception:
                        loader = TextLoader(str(path), encoding="utf-8")
                        docs = loader.load()
                elif ext in [".txt", ".md"]:
                    loader = TextLoader(str(path), encoding="utf-8")
                    docs = loader.load()
                else:
                    logger.warning(f"Unsupported file format: {ext}")
                    continue

                for doc in docs:
                    if "source" not in doc.metadata or not doc.metadata["source"]:
                        doc.metadata["source"] = path.name

                all_docs.extend(docs)
                logger.info(f"Loaded {len(docs)} pages/sections from {path.name}")

            except Exception as e:
                logger.error(f"Error loading document {path.name}: {e}")

        if not all_docs:
            return {"status": "error", "message": "No valid text extracted from uploaded files."}

        # Text Splitting
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=150)
        chunks = text_splitter.split_documents(all_docs)
        if not chunks:
            filenames = ", ".join(Path(file_path).name for file_path in file_paths)
            return {
                "status": "error",
                "message": (
                    f"No extractable text was found in {filenames}. "
                    "The PDF may be scanned or image-only; upload a text-searchable "
                    "PDF or run OCR on it first."
                )
            }

        # Vector Indexing
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vector_store.add_documents(chunks)

        # Save to disk
        self.vector_store.save_local(folder_path=str(self.vector_store_path))

        # Save metadata to SQLite DB
        for file_path in file_paths:
            p = Path(file_path)
            file_chunks = [c for c in chunks if c.metadata.get("source") == p.name]
            save_document_metadata(p.name, str(p), p.suffix, len(file_chunks))

        logger.info(f"Successfully ingested {len(file_paths)} files into FAISS index ({len(chunks)} total chunks).")
        return {
            "status": "success",
            "chunks_count": len(chunks),
            "documents": [Path(p).name for p in file_paths]
        }

    def search(self, query: str, top_k: int = 4) -> Tuple[List[Dict[str, Any]], float]:
        """
        Queries FAISS vector index and returns context chunks along with confidence score.
        Similarity distance is converted to a normalized confidence score (0.0 to 1.0).
        """
        if self.vector_store is None:
            logger.warning("Search attempted with empty vector store.")
            return [], 0.0

        try:
            # similarity_search_with_score returns L2 distance (lower score = higher similarity)
            results_with_scores = self.vector_store.similarity_search_with_score(query, k=top_k)
            
            if not results_with_scores:
                return [], 0.0

            results = []
            confidence_scores = []

            for doc, score in results_with_scores:
                # Convert L2 distance into a 0.0 - 1.0 confidence score
                # Typical L2 distances for normalized embeddings range from 0 (identical) to ~1.5+
                confidence = max(0.0, min(1.0, 1.0 - (score / 2.0)))
                confidence_scores.append(confidence)

                results.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Uploaded Document"),
                    "page": doc.metadata.get("page", 1),
                    "score": confidence
                })

            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            logger.info(f"Retriever found {len(results)} chunks for query. Avg Confidence: {avg_confidence:.2f}")
            return results, avg_confidence

        except Exception as e:
            logger.error(f"Error during FAISS similarity search: {e}")
            return [], 0.0

    def clear_index(self):
        """Clears current FAISS index and uploaded vector store."""
        self.vector_store = None
        if self.vector_store_path.exists():
            for f in self.vector_store_path.iterdir():
                try:
                    f.unlink()
                except Exception:
                    pass
        clear_documents_metadata()
        logger.info("Cleared FAISS index and document database metadata.")

    def get_status(self) -> Dict[str, Any]:
        """Returns current status of vector database."""
        docs = get_all_document_metadata()
        has_index = self.vector_store is not None
        total_chunks = sum(d["num_chunks"] for d in docs) if docs else 0
        return {
            "has_index": has_index,
            "total_documents": len(docs),
            "total_chunks": total_chunks,
            "documents": [d["filename"] for d in docs]
        }
