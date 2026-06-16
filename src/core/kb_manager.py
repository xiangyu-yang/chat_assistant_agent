import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import shutil

from src.config.settings import KNOWLEDGE_BASES_DIR

logger = logging.getLogger(__name__)


class KnowledgeBaseConfig:
    def __init__(self, name: str, description: str = "", chunk_size: int = 500, 
                 chunk_overlap: int = 50, embedding_model: str = "all-MiniLM-L6-v2",
                 rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 recall_top_k: int = 20, rerank_top_k: int = 5):
        self.name = name
        self.description = description
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.rerank_model = rerank_model
        self.recall_top_k = recall_top_k
        self.rerank_top_k = rerank_top_k
    
    def to_dict(self):
        return {
            'name': self.name,
            'description': self.description,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'embedding_model': self.embedding_model,
            'rerank_model': self.rerank_model,
            'recall_top_k': self.recall_top_k,
            'rerank_top_k': self.rerank_top_k
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            name=data.get('name', ''),
            description=data.get('description', ''),
            chunk_size=data.get('chunk_size', 500),
            chunk_overlap=data.get('chunk_overlap', 50),
            embedding_model=data.get('embedding_model', 'all-MiniLM-L6-v2'),
            rerank_model=data.get('rerank_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2'),
            recall_top_k=data.get('recall_top_k', 20),
            rerank_top_k=data.get('rerank_top_k', 5)
        )


class KBManager:
    def __init__(self, base_dir: Path = None):
        if base_dir is None:
            self.base_dir = KNOWLEDGE_BASES_DIR.parent
        else:
            self.base_dir = base_dir
        
        self.kb_dir = KNOWLEDGE_BASES_DIR
        self.kb_config_file = self.kb_dir / "kb_configs.json"
        self.kb_dir.mkdir(exist_ok=True)
        
        self._load_configs()
    
    def _load_configs(self):
        if self.kb_config_file.exists():
            try:
                with open(self.kb_config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.kb_configs = {name: KnowledgeBaseConfig.from_dict(cfg) for name, cfg in data.items()}
            except Exception as e:
                logger.error(f"Error loading KB configs: {e}")
                self.kb_configs = {}
        else:
            self.kb_configs = {}
            
            default_kb = KnowledgeBaseConfig(
                name="默认知识库",
                description="系统默认知识库，存放上传的文档",
                chunk_size=500,
                chunk_overlap=50
            )
            self.kb_configs["默认知识库"] = default_kb
            self._save_configs()
    
    def _save_configs(self):
        try:
            with open(self.kb_config_file, 'w', encoding='utf-8') as f:
                data = {name: cfg.to_dict() for name, cfg in self.kb_configs.items()}
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving KB configs: {e}")
    
    def create_kb(self, config: KnowledgeBaseConfig) -> bool:
        if config.name in self.kb_configs:
            return False
        
        self.kb_configs[config.name] = config
        
        kb_path = self.get_kb_path(config.name)
        kb_path.mkdir(parents=True, exist_ok=True)
        
        (kb_path / "uploads").mkdir(exist_ok=True)
        (kb_path / "vector_store").mkdir(exist_ok=True)
        
        self._save_configs()
        return True
    
    def delete_kb(self, name: str) -> bool:
        if name not in self.kb_configs:
            return False
        
        del self.kb_configs[name]
        self._save_configs()
        
        kb_path = self.get_kb_path(name)
        if kb_path.exists():
            shutil.rmtree(kb_path)
        
        return True
    
    def update_kb(self, name: str, config: KnowledgeBaseConfig) -> bool:
        if name not in self.kb_configs:
            return False
        
        old_path = self.get_kb_path(name)
        self.kb_configs[name] = config
        
        if config.name != name:
            new_path = self.get_kb_path(config.name)
            if old_path.exists() and not new_path.exists():
                old_path.rename(new_path)
            del self.kb_configs[name]
            self.kb_configs[config.name] = config
        
        self._save_configs()
        return True
    
    def get_kb_config(self, name: str) -> Optional[KnowledgeBaseConfig]:
        return self.kb_configs.get(name)
    
    def list_kbs(self) -> List[KnowledgeBaseConfig]:
        return list(self.kb_configs.values())
    
    def get_kb_path(self, name: str) -> Path:
        safe_name = name.replace('/', '_').replace('\\', '_').replace(':', '_')
        return self.kb_dir / safe_name
    
    def get_kb_upload_path(self, name: str) -> Path:
        return self.get_kb_path(name) / "uploads"
    
    def get_kb_vector_store_path(self, name: str) -> Path:
        return self.get_kb_path(name) / "vector_store"
    
    def get_kb_stats(self, name: str) -> Dict:
        kb_path = self.get_kb_path(name)
        vector_store_path = kb_path / "vector_store"
        
        doc_count = 0
        chunk_count = 0
        
        index_path = vector_store_path / "index.faiss"
        if index_path.exists():
            try:
                import faiss
                import pickle
                
                index = faiss.read_index(str(index_path))
                chunk_count = index.ntotal
                
                documents_path = vector_store_path / "documents.pkl"
                if documents_path.exists():
                    with open(documents_path, 'rb') as f:
                        data = pickle.load(f)
                        metadata_list = data.get('metadata', [])
                        unique_sources = set()
                        for meta in metadata_list:
                            if 'source' in meta:
                                unique_sources.add(meta['source'])
                        doc_count = len(unique_sources)
            except Exception as e:
                logger.error(f"Error loading vector store for stats: {e}")
                doc_count = 0
                chunk_count = 0
        else:
            chunk_count = 0
        
        return {
            'documents': doc_count,
            'chunks': chunk_count,
            'name': name,
            'description': self.kb_configs.get(name, KnowledgeBaseConfig(name)).description
        }