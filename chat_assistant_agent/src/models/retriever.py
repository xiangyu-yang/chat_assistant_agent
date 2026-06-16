from typing import List, Dict, Optional
import numpy as np
from src.models.embeddings import EmbeddingModel, RerankerModel
from src.data.vector_store import VectorStore
import logging

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        reranker_model: Optional[RerankerModel] = None
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model

    def recall(self, query: str, top_k: int = 20) -> List[Dict]:
        query_embedding = self.embedding_model.encode(query)
        results = self.vector_store.search(query_embedding, top_k)
        return results

    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Dict]:
        if self.reranker_model is None:
            return [{'content': doc, 'score': 1.0} for doc in documents[:top_k]]

        return self.reranker_model.rerank(query, documents, top_k)

    def retrieve(self, query: str, recall_top_k: int = 20, rerank_top_k: int = 5) -> List[Dict]:
        recalled_docs = self.recall(query, top_k=recall_top_k)

        if not recalled_docs:
            return []

        if self.reranker_model:
            documents = [doc['content'] for doc in recalled_docs]
            reranked = self.reranker_model.rerank(query, documents, top_k=rerank_top_k)

            results = []
            for item in reranked:
                original_doc = recalled_docs[item['index']]
                results.append({
                    'content': original_doc['content'],
                    'metadata': original_doc['metadata'],
                    'rerank_score': item['score'],
                    'recall_score': original_doc['distance']
                })
        else:
            results = recalled_docs[:rerank_top_k]

        return results