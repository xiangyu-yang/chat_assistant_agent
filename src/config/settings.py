import os
from pathlib import Path

os.environ['HF_ENDPOINT'] = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')

BASE_DIR = Path(__file__).parent.parent.parent
UPLOAD_DIR = Path(os.environ.get('UPLOAD_DIR', str(BASE_DIR / "uploads")))
VECTOR_STORE_DIR = Path(os.environ.get('VECTOR_STORE_DIR', str(BASE_DIR / "vector_store")))
MODELS_DIR = Path(os.environ.get('MODELS_DIR', str(BASE_DIR / "models")))
UPLOADED_FILES_LIST = Path(os.environ.get('UPLOADED_FILES_LIST', str(BASE_DIR / "uploaded_files.json")))
SESSIONS_DIR = Path(os.environ.get('SESSIONS_DIR', str(BASE_DIR / "sessions")))
KNOWLEDGE_BASES_DIR = Path(os.environ.get('KNOWLEDGE_BASES_DIR', str(BASE_DIR / "knowledge_bases")))

for dir_path in [UPLOAD_DIR, VECTOR_STORE_DIR, MODELS_DIR, SESSIONS_DIR, KNOWLEDGE_BASES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', "http://localhost:11434/v1")
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', "qwen3.6:35b-a3b-q8_0")
OLLAMA_API_KEY = os.environ.get('OLLAMA_API_KEY', "ollama")

EMBEDDING_MODEL_NAME = os.environ.get('EMBEDDING_MODEL_NAME', "BAAI/bge-m3")
RERANK_MODEL_NAME = os.environ.get('RERANK_MODEL_NAME', "BAAI/bge-reranker-v2-m3")

CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', 500))
CHUNK_OVERLAP = int(os.environ.get('CHUNK_OVERLAP', 50))

RECALL_TOP_K = int(os.environ.get('RECALL_TOP_K', 20))
RERANK_TOP_K = int(os.environ.get('RERANK_TOP_K', 5))

EMBEDDING_DIM = int(os.environ.get('EMBEDDING_DIM', 1024))