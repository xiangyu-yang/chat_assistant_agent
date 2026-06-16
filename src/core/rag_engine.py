from pathlib import Path
from typing import List, Dict, Optional, Literal
import numpy as np
from src.data.document_processor import DocumentProcessor
from src.models.embeddings import EmbeddingModel, RerankerModel
from src.data.vector_store import VectorStore
from src.models.retriever import Retriever
from src.models.llm import LLMClient
from src.config.settings import EMBEDDING_DIM
import logging

logger = logging.getLogger(__name__)

ChunkStrategy = Literal['fixed', 'hierarchical', 'semantic']


class RAGEngine:
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        rerank_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        llm_base_url: str = "http://localhost:11434/v1",
        llm_model: str = "qwen3.6:35b-a3b-q8_0",
        vector_store_path: Optional[Path] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        chunk_strategy: ChunkStrategy = 'fixed'
    ):
        self.doc_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=chunk_strategy
        )
        self.chunk_strategy = chunk_strategy

        self.embedding_model_name = embedding_model_name
        self.rerank_model_name = rerank_model_name
        
        self._embedding_model = None
        self._reranker_model = None

        self.vector_store = VectorStore(
            embedding_dim=EMBEDDING_DIM,
            store_path=vector_store_path
        )

        self._retriever = None
        self._llm_client = None
        
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
    
    @property
    def embedding_model(self):
        if self._embedding_model is None:
            logger.info("Initializing embedding model...")
            self._embedding_model = EmbeddingModel(model_name=self.embedding_model_name)
        return self._embedding_model
    
    @property
    def reranker_model(self):
        if self._reranker_model is None:
            logger.info("Initializing reranker model...")
            try:
                self._reranker_model = RerankerModel(model_name=self.rerank_model_name)
            except Exception as e:
                logger.warning(f"Failed to load reranker model: {e}. Reranking will be disabled.")
                self._reranker_model = None
        return self._reranker_model
    
    @property
    def retriever(self):
        if self._retriever is None:
            self._retriever = Retriever(
                vector_store=self.vector_store,
                embedding_model=self.embedding_model,
                reranker_model=self.reranker_model
            )
        return self._retriever
    
    @property
    def llm_client(self):
        if self._llm_client is None:
            logger.info("Initializing LLM client...")
            self._llm_client = LLMClient(
                base_url=self.llm_base_url,
                model=self.llm_model
            )
        return self._llm_client

    def update_llm_config(self, base_url: str, model: str) -> Dict:
        try:
            new_client = LLMClient(
                base_url=base_url,
                model=model
            )
            
            test_result = new_client.test_connection()
            
            if test_result["success"]:
                self.llm_client.close()
                self._llm_client = new_client
                self.llm_base_url = base_url
                self.llm_model = model
                
                logger.info(f"LLM config updated to: base_url={base_url}, model={model}")
                return {
                    "success": True,
                    "message": test_result["message"],
                    "status": "updated"
                }
            else:
                new_client.close()
                return test_result
                
        except Exception as e:
            logger.error(f"Failed to update LLM config: {e}")
            return {
                "success": False,
                "message": f"❌ 更新配置失败: {str(e)}",
                "status": "error"
            }

    def test_current_connection(self) -> Dict:
        return self.llm_client.test_connection()
    
    def test_model_connection(self, model_name: str) -> Dict:
        return self.llm_client.test_connection(model_name=model_name)

    def get_available_models(self) -> List[str]:
        return self.llm_client.get_available_models()

    def add_documents(self, file_paths: List[Path]) -> Dict:
        total_chunks = 0
        successful_files = []
        failed_files = []
        
        logger.info(f"Starting to process {len(file_paths)} files")
        
        if self._embedding_model is None:
            logger.info("Initializing embedding model...")
            self.embedding_model

        for file_path in file_paths:
            try:
                logger.info(f"Processing file: {file_path}")
                
                file_chunks = []
                for chunk in self.doc_processor.stream_chunks(file_path):
                    file_chunks.append(chunk)
                
                logger.info(f"Extracted {len(file_chunks)} chunks from {file_path.name}")

                if file_chunks:
                    documents = [
                        {
                            'content': chunk,
                            'metadata': {
                                'source': file_path.name,
                                'chunk_id': idx
                            }
                        }
                        for idx, chunk in enumerate(file_chunks)
                    ]
                    
                    embeddings = self.embedding_model.encode(
                        [doc['content'] for doc in documents],
                        batch_size=16
                    )
                    
                    self.vector_store.add_documents(embeddings, documents)
                    total_chunks += len(documents)
                    successful_files.append(file_path.name)

                    logger.info(f"Added {len(documents)} chunks from {file_path.name}")
                else:
                    logger.warning(f"No content extracted from {file_path.name}")
                    failed_files.append(file_path.name)

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}", exc_info=True)
                failed_files.append(file_path.name)

        if total_chunks > 0:
            logger.info(f"Saving vector store with {total_chunks} chunks")
            self.vector_store.save()
        else:
            logger.warning("No chunks to save, skipping vector store save")

        return {
            'total_chunks': total_chunks,
            'successful_files': successful_files,
            'failed_files': failed_files
        }
    
    def rebuild_index(self, file_paths: List[Path]) -> Dict:
        self.vector_store.reset()
        
        total_chunks = 0
        successful_files = []
        failed_files = []
        
        for file_path in file_paths:
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                failed_files.append(file_path.name)
                continue
                
            try:
                file_chunks = []
                for chunk in self.doc_processor.stream_chunks(file_path):
                    file_chunks.append(chunk)
                
                if file_chunks:
                    documents = [
                        {
                            'content': chunk,
                            'metadata': {
                                'source': file_path.name,
                                'chunk_id': idx
                            }
                        }
                        for idx, chunk in enumerate(file_chunks)
                    ]
                    
                    embeddings = self.embedding_model.encode(
                        [doc['content'] for doc in documents],
                        batch_size=16
                    )
                    
                    self.vector_store.add_documents(embeddings, documents)
                    total_chunks += len(documents)
                    successful_files.append(file_path.name)
                    
                    logger.info(f"Rebuilt {len(documents)} chunks from {file_path.name}")
                else:
                    failed_files.append(file_path.name)
                    
            except Exception as e:
                logger.error(f"Error rebuilding {file_path}: {e}")
                failed_files.append(file_path.name)
        
        self.vector_store.save()
        self.vector_store._loaded = True
        
        logger.info(f"Rebuilt {total_chunks} chunks total")
        
        return {
            'total_chunks': total_chunks,
            'successful_files': successful_files,
            'failed_files': failed_files
        }

    def query_stream(
        self,
        question: str,
        recall_top_k: int = 20,
        rerank_top_k: int = 5,
        use_rerank: bool = True,
        feedback_history: Optional[List[Dict]] = None
    ):
        stats = self.vector_store.get_stats()
        has_knowledge_base = stats['total_documents'] > 0

        if has_knowledge_base:
            retrieved_docs = self.retriever.retrieve(
                query=question,
                recall_top_k=recall_top_k,
                rerank_top_k=rerank_top_k if use_rerank else recall_top_k
            )

            if retrieved_docs:
                contexts = [doc['content'] for doc in retrieved_docs]
                yield {
                    'type': 'sources',
                    'data': [
                        {
                            'content': doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'],
                            'source': doc['metadata']['source'],
                            'score': doc.get('rerank_score', doc.get('distance', 0))
                        }
                        for doc in retrieved_docs
                    ],
                    'contexts': contexts,
                    'mode': 'rag'
                }
                
                for chunk in self.llm_client.generate_with_context_stream(
                    query=question,
                    context=contexts,
                    feedback_history=feedback_history
                ):
                    yield {
                        'type': 'answer_chunk',
                        'data': chunk,
                        'mode': 'rag'
                    }
                return

        yield {
            'type': 'sources',
            'data': [],
            'contexts': [],
            'mode': 'chat'
        }
        
        for chunk in self.llm_client.chat_stream(query=question, feedback_history=feedback_history):
            yield {
                'type': 'answer_chunk',
                'data': chunk,
                'mode': 'chat'
            }
    
    def query(
        self,
        question: str,
        recall_top_k: int = 20,
        rerank_top_k: int = 5,
        use_rerank: bool = True
    ) -> Dict:
        stats = self.vector_store.get_stats()
        has_knowledge_base = stats['total_documents'] > 0
        
        thinking_steps = []

        if has_knowledge_base:
            thinking_steps.append(f"🧠 分析问题: {question}")
            thinking_steps.append(f"📚 知识库中有 {stats['total_documents']} 个文档，{stats['index_size']} 个片段")
            thinking_steps.append(f"🔍 正在检索相关文档... (Recall: top-{recall_top_k})")
            
            retrieved_docs = self.retriever.retrieve(
                query=question,
                recall_top_k=recall_top_k,
                rerank_top_k=rerank_top_k if use_rerank else recall_top_k
            )

            if retrieved_docs:
                if use_rerank:
                    thinking_steps.append(f"⚡ 正在进行重排... (ReRank: top-{rerank_top_k})")
                thinking_steps.append(f"✅ 检索到 {len(retrieved_docs)} 个相关片段")
                thinking_steps.append(f"📖 正在阅读文档并生成回答...")
                
                contexts = [doc['content'] for doc in retrieved_docs]
                answer = self.llm_client.generate_with_context(
                    query=question,
                    context=contexts
                )

                return {
                    'answer': answer,
                    'sources': [
                        {
                            'content': doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'],
                            'source': doc['metadata']['source'],
                            'score': doc.get('rerank_score', doc.get('distance', 0))
                        }
                        for doc in retrieved_docs
                    ],
                    'contexts': contexts,
                    'mode': 'rag',
                    'thinking': thinking_steps
                }
            else:
                thinking_steps.append("⚠️ 知识库中未找到相关文档，切换到对话模式")

        thinking_steps.append("💬 使用大模型直接回答...")
        answer = self.llm_client.chat(query=question)

        return {
            'answer': answer,
            'sources': [],
            'contexts': [],
            'mode': 'chat',
            'thinking': thinking_steps
        }
    
    def get_stats(self) -> Dict:
        return self.vector_store.get_stats()

    def reset(self):
        self.vector_store.index = None
        self.vector_store.documents = []
        self.vector_store.metadata = []
        self.vector_store.save()