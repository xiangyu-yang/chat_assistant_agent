import numpy as np
from typing import List, Union
import logging
import os

logger = logging.getLogger(__name__)

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import torch
torch.set_default_device('cpu')
torch.set_num_threads(1)
os.environ['TRANSFORMERS_DEVICE'] = 'cpu'

logger.info("Environment configured for CPU")

from sentence_transformers import SentenceTransformer, CrossEncoder
logger.info("sentence-transformers imported successfully")


class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._load_model()

    def _load_model(self):
        logger.info(f"Loading embedding model: {self.model_name}")
        
        import torch
        device = 'cpu'
        logger.info(f"Forcing CPU device for stability")
        
        self._model = SentenceTransformer(
            self.model_name, 
            device=device,
            cache_folder=None
        )
        self._model.max_seq_length = 512
        logger.info("Embedding model loaded successfully")

    def encode(self, texts: Union[str, List[str]], normalize: bool = True, batch_size: int = 32) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        total_texts = len(texts)
        logger.info(f"Starting encoding {total_texts} texts with batch_size={batch_size}")
        
        all_embeddings = []
        total_batches = (total_texts + batch_size - 1) // batch_size
        
        import os
        original_num_threads = os.environ.get('OMP_NUM_THREADS')
        os.environ['OMP_NUM_THREADS'] = '1'
        
        try:
            for i in range(0, total_texts, batch_size):
                batch = texts[i:i+batch_size]
                logger.info(f"Encoding batch {i//batch_size + 1}/{total_batches}: {len(batch)} texts")
                
                batch_embeddings = self._model.encode(
                    batch, 
                    convert_to_numpy=True, 
                    show_progress_bar=False,
                    batch_size=batch_size,
                    device='cpu'
                )
                all_embeddings.append(batch_embeddings)
                logger.info(f"Batch {i//batch_size + 1} encoded successfully")
        finally:
            if original_num_threads is not None:
                os.environ['OMP_NUM_THREADS'] = original_num_threads
        
        if len(all_embeddings) == 1:
            embeddings = all_embeddings[0]
        else:
            embeddings = np.vstack(all_embeddings)

        if normalize and embeddings.shape[0] > 0:
            norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norm + 1e-10)

        logger.info(f"Encoding completed: {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
        return embeddings

    def get_dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()


class RerankerModel:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._available = False
        self._load_model()

    def _load_model(self):
        try:
            logger.info(f"Loading reranker model: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
            self._available = True
            logger.info("Reranker model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load reranker model: {e}")
            logger.warning("Reranking will be disabled")

    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[dict]:
        if not self._available:
            return [{'index': i, 'document': doc, 'score': 1.0} for i, doc in enumerate(documents[:top_k])]

        try:
            scores = self._model.predict([(query, doc) for doc in documents])
            scored_docs = [(i, doc, float(score)) for i, (doc, score) in enumerate(zip(documents, scores))]
            scored_docs.sort(key=lambda x: x[2], reverse=True)
            
            return [{'index': i, 'document': doc, 'score': score} for i, doc, score in scored_docs[:top_k]]
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            return [{'index': i, 'document': doc, 'score': 1.0} for i, doc in enumerate(documents[:top_k])]
