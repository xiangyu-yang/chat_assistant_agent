import faiss
import numpy as np
from typing import List, Dict, Optional
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, embedding_dim: int, store_path: Optional[Path] = None):
        self.embedding_dim = embedding_dim
        self.store_path = store_path
        self.index = None
        self.documents = []
        self.metadata = []
        self._loaded = False

    def _ensure_loaded(self):
        if not self._loaded and self.store_path and (self.store_path / "index.faiss").exists():
            self.load()
            self._loaded = True

    def add_documents(self, embeddings: np.ndarray, documents: List[Dict]):
        if len(embeddings) == 0:
            return

        if self.index is None:
            self.index = faiss.IndexFlatIP(self.embedding_dim)

        embeddings = embeddings.astype('float32')
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

        self.documents.extend([doc['content'] for doc in documents])
        self.metadata.extend([doc['metadata'] for doc in documents])
    
    def reset(self):
        self.index = None
        self.documents = []
        self.metadata = []
        self._loaded = False
        logger.info("Vector store reset successfully")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        self._ensure_loaded()
        if self.index is None or self.index.ntotal == 0:
            return []

        query_embedding = query_embedding.astype('float32')
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        faiss.normalize_L2(query_embedding)

        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                results.append({
                    'content': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'distance': float(distance)
                })

        return results

    def save(self):
        if self.store_path is None:
            logger.error("Cannot save vector store: store_path is None")
            raise ValueError("Vector store path is not set")

        self.store_path.mkdir(parents=True, exist_ok=True)

        if self.index is not None:
            faiss.write_index(
                self.index,
                str(self.store_path / "index.faiss")
            )

        with open(self.store_path / "documents.pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata
            }, f)

        logger.info(f"Vector store saved to {self.store_path}")

    def load(self):
        if self.store_path is None:
            return

        try:
            self.index = faiss.read_index(str(self.store_path / "index.faiss"))

            with open(self.store_path / "documents.pkl", 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.metadata = data['metadata']

            logger.info(f"Vector store loaded from {self.store_path}")
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            self.index = None
            self.documents = []
            self.metadata = []

    def get_stats(self) -> Dict:
        self._ensure_loaded()
        unique_sources = set()
        for meta in self.metadata:
            if 'source' in meta:
                unique_sources.add(meta['source'])
        
        return {
            'total_documents': len(unique_sources),
            'embedding_dim': self.embedding_dim,
            'index_size': self.index.ntotal if self.index else 0
        }
    
    def get_all_documents(self) -> List[str]:
        self._ensure_loaded()
        unique_sources = set()
        for meta in self.metadata:
            if 'source' in meta:
                unique_sources.add(meta['source'])
        return sorted(list(unique_sources))
    
    def get_chunks_by_document(self, document_name: str) -> List[Dict]:
        self._ensure_loaded()
        chunks = []
        for i, meta in enumerate(self.metadata):
            if 'source' in meta and meta['source'] == document_name:
                chunks.append({
                    'content': self.documents[i],
                    'metadata': meta,
                    'index': i
                })
        return chunks
    
    def get_all_chunks(self) -> List[Dict]:
        self._ensure_loaded()
        chunks = []
        for i, doc in enumerate(self.documents):
            chunks.append({
                'content': doc,
                'metadata': self.metadata[i],
                'index': i
            })
        return chunks
    
    def __enter__(self):
        self._ensure_loaded()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.save()
        return False