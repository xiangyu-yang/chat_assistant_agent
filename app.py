import streamlit as st
import os
import json
import time
import threading
import queue
import logging
from pathlib import Path
from src.config.settings import (
    UPLOADED_FILES_LIST,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    EMBEDDING_MODEL_NAME,
    RERANK_MODEL_NAME
)
from src.services.search import WebSearch
from src.core.kb_manager import KBManager, KnowledgeBaseConfig
from src.utils.session_manager import (
    load_sessions,
    load_session,
    save_session,
    delete_session,
    generate_session_id
)
from src.utils.file_utils import (
    load_uploaded_files as utils_load_uploaded_files,
    save_uploaded_files as utils_save_uploaded_files
)
from src.services.kkfileview_service import is_kkfileview_available, get_file_preview_url
from src.data.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)


def render_text_preview(file_path: str, title: str = "文本预览") -> None:
    """Render text preview for a file"""
    st.markdown(f"### {title}")
    try:
        processor = DocumentProcessor()
        content = processor.load_document(file_path)
        if content:
            st.text_area("", content, height=600)
        else:
            st.warning("该文件没有可提取的文本内容")
    except Exception as e:
        st.error(f"加载文件时出错: {str(e)}")


def render_llm_config_section():
    st.markdown("### 🔧 LLM 配置")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        new_base_url = st.text_input(
            "服务地址",
            value=st.session_state.get('llm_base_url', OLLAMA_BASE_URL),
            help="大模型API服务地址"
        )
    
    with col2:
        try:
            available_models = st.session_state.rag_engine.get_available_models()
            current_model = st.session_state.get('llm_model', OLLAMA_MODEL)
            
            if current_model not in available_models:
                available_models.insert(0, current_model)
            
            new_model = st.selectbox(
                "选择模型",
                options=available_models if available_models else [current_model],
                index=available_models.index(current_model) if (available_models and current_model in available_models) else 0,
                help="从Ollama获取的可用模型列表"
            )
        except Exception:
            new_model = st.text_input(
                "模型名称",
                value=st.session_state.get('llm_model', OLLAMA_MODEL)
            )
    
    st.markdown("---")
    col_info, col_status = st.columns([3, 1])
    with col_info:
        st.markdown(f"""
        <div style="font-size: 0.9rem;">
        <strong>已应用配置：</strong>
        <br>服务地址: <code>{st.session_state.get('llm_base_url', OLLAMA_BASE_URL)}</code>
        <br>模型名称: <code>{st.session_state.get('llm_model', OLLAMA_MODEL)}</code>
        </div>
        """, unsafe_allow_html=True)
    
    with col_status:
        if st.session_state.last_test_result:
            if st.session_state.last_test_result["success"]:
                st.success("✓ 已连接")
            else:
                st.error("✗ 未连接")
    
    if st.session_state.last_test_result:
        if st.session_state.last_test_result["success"]:
            st.success(st.session_state.last_test_result["message"])
        else:
            st.error(st.session_state.last_test_result["message"])
    
    col1, col2 = st.columns([1, 1])
    with col1:
        test_button = st.button("🔌 测试连接", use_container_width=True)
        if test_button:
            with st.spinner("测试中..."):
                result = st.session_state.rag_engine.test_model_connection(model_name=new_model)
                st.session_state.last_test_result = result
                st.rerun()
    
    with col2:
        apply_button = st.button("💾 应用配置", use_container_width=True)
        if apply_button:
            with st.spinner("应用中..."):
                result = st.session_state.rag_engine.update_llm_config(
                    base_url=new_base_url,
                    model=new_model
                )
                st.session_state.last_test_result = result
                if result["success"]:
                    st.session_state.llm_base_url = new_base_url
                    st.session_state.llm_model = new_model
                st.rerun()


def render_kb_manager_page():
    st.title("🏛️ 知识库管理")
    
    if st.button("← 返回主页面", key="back_from_kb_manager"):
        st.session_state.page = 'main'
        st.rerun()
    
    st.markdown("---")
    
    if 'kb_manager' not in st.session_state:
        st.session_state.kb_manager = KBManager(Path(__file__).parent)
    
    kb_manager = st.session_state.kb_manager
    
    with st.expander("➕ 新建知识库", expanded=False):
        st.markdown("#### 创建新的知识库")
        
        new_kb_name = st.text_input("知识库名称", placeholder="请输入知识库名称")
        new_kb_desc = st.text_area("知识库描述", placeholder="请输入知识库描述（可选）", height=60)
        
        st.markdown("**文档处理配置:**")
        col1, col2 = st.columns(2)
        with col1:
            new_chunk_size = st.number_input("片段大小", min_value=100, max_value=2000, value=500, step=50)
        with col2:
            new_chunk_overlap = st.number_input("重叠大小", min_value=0, max_value=200, value=50, step=10)
        
        st.markdown("**检索配置:**")
        col3, col4 = st.columns(2)
        with col3:
            new_recall_k = st.number_input("召回数量", min_value=1, max_value=50, value=20)
        with col4:
            new_rerank_k = st.number_input("重排数量", min_value=1, max_value=20, value=5)
        
        if st.button("✨ 创建知识库", use_container_width=True):
            if not new_kb_name.strip():
                st.error("请输入知识库名称")
            else:
                config = KnowledgeBaseConfig(
                    name=new_kb_name.strip(),
                    description=new_kb_desc.strip(),
                    chunk_size=new_chunk_size,
                    chunk_overlap=new_chunk_overlap,
                    recall_top_k=new_recall_k,
                    rerank_top_k=new_rerank_k
                )
                success = kb_manager.create_kb(config)
                if success:
                    st.success(f"✅ 知识库 '{new_kb_name}' 创建成功！")
                    st.rerun()
                else:
                    st.error(f"❌ 知识库 '{new_kb_name}' 已存在")
    
    st.markdown("---")
    
    st.markdown("#### 📋 知识库列表")
    kbs = kb_manager.list_kbs()
    
    if not kbs:
        st.info("暂无知识库，请创建一个新的知识库")
        return
    
    for kb in kbs:
        stats = kb_manager.get_kb_stats(kb.name)
        is_active = st.session_state.get('current_kb') == kb.name
        
        with st.container():
            col_header = st.columns([3, 1, 1])
            with col_header[0]:
                st.markdown(f"**📁 {kb.name}**")
                if kb.description:
                    st.markdown(f"<small style='color: #666'>{kb.description}</small>", unsafe_allow_html=True)
            
            with col_header[1]:
                if st.button("⚙️ 配置", key=f"config_{kb.name}", use_container_width=True):
                    st.session_state.editing_kb = kb.name
                    st.rerun()
            
            with col_header[2]:
                if kb.name != "默认知识库":
                    if st.button("🗑️ 删除", key=f"delete_kb_{kb.name}", use_container_width=True):
                        st.session_state.deleting_kb = kb.name
            
            col_stats = st.columns(3)
            with col_stats[0]:
                st.metric("文档数", stats['documents'])
            with col_stats[1]:
                st.metric("片段数", stats['chunks'])
            with col_stats[2]:
                if st.button("🎯 使用此知识库", key=f"use_{kb.name}", use_container_width=True, type="primary" if is_active else "secondary"):
                    st.session_state.current_kb = kb.name
                    from src.core.rag_engine import RAGEngine
                    vector_store_path = kb_manager.get_kb_vector_store_path(kb.name)
                    st.session_state.rag_engine = RAGEngine(
                            vector_store_path=vector_store_path,
                            embedding_model_name=EMBEDDING_MODEL_NAME,
                            rerank_model_name=RERANK_MODEL_NAME
                        )
                    if 'last_kb' in st.session_state:
                        del st.session_state.last_kb
                    st.success(f"✅ 已切换到知识库 '{kb.name}'")
                    st.rerun()
            
            if st.session_state.get('deleting_kb') == kb.name:
                st.warning(f"⚠️ 确定要删除知识库 '{kb.name}' 吗？此操作将删除所有文档和配置，不可恢复！")
                col_confirm = st.columns(2)
                with col_confirm[0]:
                    if st.button("✅ 确认删除", key=f"confirm_delete_kb_{kb.name}"):
                        success = kb_manager.delete_kb(kb.name)
                        if success:
                            st.success(f"✅ 知识库 '{kb.name}' 已删除")
                            if st.session_state.get('current_kb') == kb.name:
                                st.session_state.current_kb = "默认知识库"
                            st.session_state.deleting_kb = None
                            st.rerun()
                        else:
                            st.error("删除失败")
                with col_confirm[1]:
                    if st.button("❌ 取消", key=f"cancel_delete_kb_{kb.name}"):
                        st.session_state.deleting_kb = None
                        st.rerun()
            
            if st.session_state.get('editing_kb') == kb.name:
                st.markdown("---")
                st.markdown(f"#### ⚙️ 编辑知识库: {kb.name}")
                
                new_desc = st.text_area("描述", value=kb.description, height=60, key=f"desc_{kb.name}")
                col_edit = st.columns(2)
                with col_edit[0]:
                    edit_chunk_size = st.number_input("片段大小", min_value=100, max_value=2000, value=kb.chunk_size, step=50, key=f"chunk_size_{kb.name}")
                with col_edit[1]:
                    edit_chunk_overlap = st.number_input("重叠大小", min_value=0, max_value=200, value=kb.chunk_overlap, step=10, key=f"chunk_overlap_{kb.name}")
                
                col_edit2 = st.columns(2)
                with col_edit2[0]:
                    edit_recall_k = st.number_input("召回数量", min_value=1, max_value=50, value=kb.recall_top_k, key=f"recall_k_{kb.name}")
                with col_edit2[1]:
                    edit_rerank_k = st.number_input("重排数量", min_value=1, max_value=20, value=kb.rerank_top_k, key=f"rerank_k_{kb.name}")
                
                col_buttons = st.columns(2)
                with col_buttons[0]:
                    if st.button("💾 保存配置", key=f"save_config_{kb.name}", use_container_width=True):
                        updated_config = KnowledgeBaseConfig(
                            name=kb.name,
                            description=new_desc,
                            chunk_size=edit_chunk_size,
                            chunk_overlap=edit_chunk_overlap,
                            recall_top_k=edit_recall_k,
                            rerank_top_k=edit_rerank_k
                        )
                        success = kb_manager.update_kb(kb.name, updated_config)
                        if success:
                            st.success("✅ 配置已保存")
                            st.session_state.editing_kb = None
                            st.rerun()
                        else:
                            st.error("保存失败")
                with col_buttons[1]:
                    if st.button("❌ 取消", key=f"cancel_edit_{kb.name}", use_container_width=True):
                        st.session_state.editing_kb = None
                        st.rerun()
            
            st.markdown("---")


def get_current_kb_paths():
    current_kb = st.session_state.get('current_kb', '默认知识库')
    
    if 'kb_manager' in st.session_state:
        kb_manager = st.session_state.kb_manager
        upload_path = kb_manager.get_kb_upload_path(current_kb)
        vector_store_path = kb_manager.get_kb_vector_store_path(current_kb)
        uploaded_files_path = upload_path.parent / "uploaded_files.json"
    else:
        from src.config.settings import UPLOAD_DIR, VECTOR_STORE_DIR
        upload_path = UPLOAD_DIR
        vector_store_path = VECTOR_STORE_DIR
        uploaded_files_path = UPLOADED_FILES_LIST
    
    return upload_path, vector_store_path, uploaded_files_path


def load_kb_uploaded_files(uploaded_files_path):
    return utils_load_uploaded_files(uploaded_files_path)


def save_kb_uploaded_files(uploaded_files_path, files_list):
    utils_save_uploaded_files(uploaded_files_path, files_list)


def render_knowledge_base_section():
    st.markdown("### 📚 知识库管理")
    
    if st.button("🏛️ 管理知识库", use_container_width=True, type="secondary"):
        st.session_state.page = 'kb_manager'
        st.rerun()
    
    current_kb = st.session_state.get('current_kb', '默认知识库')
    st.markdown(f"<small style='color: #666'>当前知识库: <strong>{current_kb}</strong></small>", unsafe_allow_html=True)
    st.markdown("---")
    
    upload_path, vector_store_path, uploaded_files_path = get_current_kb_paths()
    
    if 'uploaded_files' not in st.session_state or st.session_state.get('last_kb') != current_kb:
        st.session_state.uploaded_files = load_kb_uploaded_files(uploaded_files_path)
        st.session_state.last_kb = current_kb
    
    uploaded_files = st.file_uploader(
        "上传文档",
        type=['pdf', 'docx', 'txt', 'xlsx', 'xls'],
        accept_multiple_files=True,
        help="支持 PDF、DOCX、TXT、Excel 格式"
    )
    
    with st.expander("⚙️ 分片策略设置", expanded=False):
        chunk_strategy = st.selectbox(
            "选择分片策略",
            options=['fixed', 'hierarchical', 'semantic'],
            format_func=lambda x: {
                'fixed': '📏 固定长度分片',
                'hierarchical': '📊 父子层次分片',
                'semantic': '🧠 语义向量分片'
            }[x],
            index=['fixed', 'hierarchical', 'semantic'].index(st.session_state.get('chunk_strategy', 'fixed')),
            help="选择文档分片的策略"
        )
        st.session_state.chunk_strategy = chunk_strategy
        
        if chunk_strategy == 'fixed':
            st.markdown("**固定长度分片**：按固定字符数分割文档，保持简单高效。适合结构化文档。")
        elif chunk_strategy == 'hierarchical':
            st.markdown("**父子层次分片**：先按章节标题分割，再在章节内进行固定长度分片。适合有明确章节结构的文档。")
        elif chunk_strategy == 'semantic':
            st.markdown("**语义向量分片**：按句子边界分割，保持语义完整性。适合非结构化文本。")
        
        chunk_size = st.slider(
            "分片大小（字符数）",
            min_value=100,
            max_value=2000,
            value=st.session_state.get('chunk_size', 500),
            step=100,
            help="每个分片的最大字符数"
        )
        st.session_state.chunk_size = chunk_size
        
        chunk_overlap = st.slider(
            "重叠大小（字符数）",
            min_value=0,
            max_value=200,
            value=st.session_state.get('chunk_overlap', 50),
            step=10,
            help="相邻分片之间的重叠字符数"
        )
        st.session_state.chunk_overlap = chunk_overlap
    
    if uploaded_files:
        if st.button("✨ 处理文档", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("📁 正在保存上传文件...")
                upload_path.mkdir(parents=True, exist_ok=True)
                
                saved_files = []
                duplicate_files = []
                new_files = []
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    if uploaded_file.name in st.session_state.uploaded_files:
                        duplicate_files.append(uploaded_file.name)
                    else:
                        save_path = upload_path / uploaded_file.name
                        with open(save_path, 'wb') as f:
                            f.write(uploaded_file.getbuffer())
                        saved_files.append(save_path)
                        new_files.append(uploaded_file.name)
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                if duplicate_files:
                    st.warning(f"⚠️ 以下文件已存在，已跳过：{', '.join(duplicate_files)}")
                
                if not saved_files:
                    status_text.text("")
                    progress_bar.empty()
                    st.info("💡 所有选择的文件都已存在，请选择其他文件上传")
                    return
                
                current_strategy = st.session_state.get('chunk_strategy', 'fixed')
                current_chunk_size = st.session_state.get('chunk_size', 500)
                current_chunk_overlap = st.session_state.get('chunk_overlap', 50)
                
                if (st.session_state.rag_engine.chunk_strategy != current_strategy or
                    st.session_state.rag_engine.doc_processor.chunk_size != current_chunk_size or
                    st.session_state.rag_engine.doc_processor.chunk_overlap != current_chunk_overlap):
                    status_text.text("🔄 正在重新初始化RAG引擎（加载嵌入模型...）")
                    from src.core.rag_engine import RAGEngine
                    from src.config.settings import EMBEDDING_MODEL_NAME, RERANK_MODEL_NAME
                    
                    st.session_state.rag_engine = RAGEngine(
                        vector_store_path=vector_store_path,
                        embedding_model_name=EMBEDDING_MODEL_NAME,
                        rerank_model_name=RERANK_MODEL_NAME,
                        chunk_size=current_chunk_size,
                        chunk_overlap=current_chunk_overlap,
                        chunk_strategy=current_strategy
                    )
                
                status_text.text("🧠 正在处理文档并生成向量...")
                progress_bar.progress(0.5)
                
                result = st.session_state.rag_engine.add_documents(saved_files)
                
                progress_bar.progress(1.0)
                status_text.text("✅ 处理完成！")
                
                st.session_state.uploaded_files.extend(new_files)
                save_kb_uploaded_files(uploaded_files_path, st.session_state.uploaded_files)

                st.success(f"""
                ✅ 文档处理完成！
                - 成功处理: {len(result['successful_files'])} 个文件
                - 生成片段: {result['total_chunks']} 个
                """)
                if result['failed_files']:
                    st.warning(f"失败文件: {', '.join(result['failed_files'])}")
                
                st.rerun()

            except Exception as e:
                status_text.text("❌ 处理失败")
                st.error(f"处理文档时出错: {str(e)}")
            finally:
                progress_bar.empty()
                status_text.empty()
    
    if st.session_state.get('uploaded_files') or upload_path.exists():
        actual_files = []
        if upload_path.exists():
            actual_files = [f.name for f in upload_path.iterdir() if f.is_file() and not f.name.startswith('.')]
        
        with st.expander(f"📁 已上传文件 ({len(actual_files)})", expanded=False):
            if 'file_to_delete' not in st.session_state:
                st.session_state.file_to_delete = None
            
            display_files = actual_files[-10:]
            dynamic_height = min(len(display_files) * 35 + 20, 200)
            
            st.markdown(f"""
            <style>
            .file-list-container {{
                max-height: {dynamic_height}px;
                overflow-y: auto;
            }}
            </style>
            """, unsafe_allow_html=True)
            
            if st.session_state.file_to_delete:
                st.warning(f"⚠️ 确定要彻底删除文件 '{st.session_state.file_to_delete}' 吗？此操作不可恢复！")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ 确认删除", key="confirm_delete"):
                        try:
                            file_path = upload_path / st.session_state.file_to_delete
                            if file_path.exists():
                                file_path.unlink()
                            
                            if st.session_state.file_to_delete in st.session_state.uploaded_files:
                                st.session_state.uploaded_files.remove(st.session_state.file_to_delete)
                            
                            remaining_files = [upload_path / f for f in actual_files if f != st.session_state.file_to_delete and (upload_path / f).exists()]
                            st.session_state.rag_engine.rebuild_index(remaining_files)
                            
                            save_kb_uploaded_files(uploaded_files_path, st.session_state.uploaded_files)
                            
                            st.success(f"✅ 文件 '{st.session_state.file_to_delete}' 已彻底删除")
                            st.session_state.file_to_delete = None
                            st.rerun()
                        except Exception as e:
                            st.error(f"删除文件时出错: {str(e)}")
                            st.session_state.file_to_delete = None
                
                with col2:
                    if st.button("❌ 取消", key="cancel_delete"):
                        st.session_state.file_to_delete = None
                        st.rerun()
            else:
                with st.container():
                    st.markdown('<div class="file-list-container">', unsafe_allow_html=True)
                    
                    for idx, filename in enumerate(display_files):
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            if st.button(f"📄 {filename}", key=f"view_file_{idx}_{len(actual_files)}", help=f"查看 {filename}", use_container_width=True):
                                st.session_state.page = 'file_preview'
                                st.session_state.preview_file = filename
                                st.rerun()
                        with col2:
                            if st.button("🗑️", key=f"delete_btn_{idx}_{len(actual_files)}", help=f"删除 {filename}"):
                                st.session_state.file_to_delete = filename
                                st.rerun()
                    
                    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### ⚙️ 检索设置")
    col1, col2 = st.columns([1, 1])
    with col1:
        recall_k = st.slider("召回数量", 5, 50, 20, help="Recall阶段返回的文档数量")
    with col2:
        rerank_k = st.slider("重排数量", 1, 10, 5, help="Rerank后返回的文档数量")
    
    use_rerank = st.checkbox("启用重排", value=True, help="是否启用Rerank阶段")
    
    st.session_state.recall_k = recall_k
    st.session_state.rerank_k = rerank_k
    st.session_state.use_rerank = use_rerank
    
    st.markdown("---")
    st.markdown("#### 📊 知识库统计")
    
    if 'kb_manager' in st.session_state:
        kb_stats = st.session_state.kb_manager.get_kb_stats(current_kb)
        rag_stats = st.session_state.rag_engine.get_stats()
        
        stats = {
            'total_documents': kb_stats['documents'],
            'index_size': rag_stats['index_size']
        }
    else:
        stats = st.session_state.rag_engine.get_stats()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("文档数", stats['total_documents'])
    
    with col2:
        st.metric("片段数", stats['index_size'])
    
    if stats['total_documents'] > 0 or stats['index_size'] > 0:
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if stats['total_documents'] > 0:
                if st.button("📋 文档列表", key="doc-btn", use_container_width=True):
                    st.session_state.page = 'doc_preview'
                    st.rerun()
        
        with col_btn2:
            if stats['index_size'] > 0:
                if st.button("📋 片段列表", key="chunk-btn", use_container_width=True):
                    st.session_state.page = 'chunk_preview'
                    st.rerun()

    if stats['total_documents'] == 0:
        st.info("💡 当前知识库为空，系统将以对话模式运行")

    if st.button("🗑️ 清空知识库", use_container_width=True):
        try:
            upload_path, vector_store_path, uploaded_files_path = get_current_kb_paths()
            
            if upload_path.exists():
                for file_path in upload_path.glob('*'):
                    if file_path.is_file():
                        file_path.unlink()
            
            if vector_store_path.exists():
                for file_path in vector_store_path.glob('*'):
                    if file_path.is_file():
                        file_path.unlink()
            
            st.session_state.rag_engine.reset()
            
            st.session_state.uploaded_files = []
            st.session_state.chat_history = []
            
            save_kb_uploaded_files(uploaded_files_path, [])
            
            st.success("✅ 知识库已彻底清空，所有文档和片段均已删除")
            
            st.rerun()
        except Exception as e:
            st.error(f"清空知识库时出错: {str(e)}")


def render_chat_interface():
    feedback_history = []
    for i, msg in enumerate(st.session_state.chat_history):
        if msg.get('role') == 'assistant' and i > 0:
            user_msg = st.session_state.chat_history[i - 1]
            if user_msg.get('role') == 'user':
                likes = msg.get('likes', 0)
                dislikes = msg.get('dislikes', 0)
                if likes > 0 or dislikes > 0:
                    feedback_history.append({
                        'query': user_msg.get('content', ''),
                        'response': msg.get('content', ''),
                        'rating': 1 if likes > dislikes else -1
                    })
    
    if feedback_history:
        display_feedback = feedback_history[-5:]
        with st.expander(f"🔍 当前反馈历史 ({len(display_feedback)} 条，最多显示5条)", expanded=False):
            for idx, feedback in enumerate(display_feedback):
                rating_icon = "❤️" if feedback['rating'] > 0 else "💔"
                rating_text = "好评" if feedback['rating'] > 0 else "改进"
                st.markdown(f"**{rating_icon} {rating_text} {idx + 1}:**")
                st.markdown(f"**用户问:** {feedback['query'][: 50]}...")
                st.markdown(f"**回答:** {feedback['response'][: 100]}...")
                st.divider()
    
    for idx, message in enumerate(st.session_state.chat_history):
        if message['role'] == 'user':
            with st.chat_message("user"):
                st.markdown(message['content'])
        else:
            with st.chat_message("assistant"):
                thinking_content = message.get('thinking', '')
                answer_text = message.get('content', '')
                
                if thinking_content:
                    with st.expander("🧠 思考过程", expanded=False):
                        st.markdown(thinking_content)
                
                st.markdown(answer_text)
                
                if message.get('sources'):
                    with st.expander("📚 查看参考来源", expanded=False):
                        for src_idx, source in enumerate(message['sources'], 1):
                            score_type = "相关度" if 'rerank_score' in source else "相似度"
                            score = source.get('rerank_score', source.get('distance', 0))
                            source_type = source.get('type', 'knowledge')
                            
                            if source_type == 'web' and source.get('url'):
                                st.markdown(f"""
                                <div class="source-card">
                                    <strong>🌐 搜索结果 {src_idx}:</strong> <a href="{source['url']}" target="_blank">{source['source']}</a> ({score_type}: {score:.4f})<br>
                                    <em>{source['content']}</em>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="source-card">
                                    <strong>📄 来源 {src_idx}:</strong> {source['source']} ({score_type}: {score:.4f})<br>
                                    <em>{source['content']}</em>
                                </div>
                                """, unsafe_allow_html=True)
                
                message_key = f"msg_{idx}"
                message_likes = message.get('likes', 0)
                message_dislikes = message.get('dislikes', 0)
                is_liked = st.session_state.get(f"liked_{message_key}", message_likes > 0)
                is_disliked = st.session_state.get(f"disliked_{message_key}", message_dislikes > 0)
                copied = st.session_state.get(f"copied_{message_key}", False)
                
                st.markdown("""
                <style>
                .action-btn {
                    background: transparent;
                    border: none;
                    cursor: pointer;
                    padding: 8px 12px;
                    border-radius: 8px;
                    transition: all 0.2s;
                    font-size: 16px;
                }}
                .action-btn:hover {{
                    background: rgba(0,0,0,0.05);
                }}
                .action-btn:active {{
                    transform: scale(0.95);
                }}
                </style>
                """, unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns([0.8, 0.8, 0.8, 5])
                with col1:
                    copy_icon = "✓" if copied else "📋"
                    copy_color = "#22c55e" if copied else "#6b7280"
                    copy_button_html = f"""
                    <button 
                        id="copy-btn-{message_key}"
                        onclick="copyToClipboard_{message_key}()"
                        style="
                            width: 100%;
                            padding: 6px 12px;
                            border: none;
                            border-radius: 8px;
                            cursor: pointer;
                            font-size: 18px;
                            background-color: transparent;
                            color: {copy_color};
                            transition: all 0.2s;
                        "
                        onmouseover="this.style.opacity='0.7'"
                        onmouseout="this.style.opacity='1'"
                        title="复制回答"
                    >
                        {copy_icon}
                    </button>
                    <script>
                    function copyToClipboard_{message_key}() {{
                        const text = {json.dumps(answer_text)};
                        navigator.clipboard.writeText(text).then(function() {{
                            const btn = document.getElementById('copy-btn-{message_key}');
                            btn.innerHTML = '✓';
                            btn.style.color = '#22c55e';
                        }}).catch(function(err) {{
                            console.error('Failed to copy:', err);
                            const textarea = document.createElement('textarea');
                            textarea.value = text;
                            document.body.appendChild(textarea);
                            textarea.select();
                            try {{
                                document.execCommand('copy');
                                const btn = document.getElementById('copy-btn-{message_key}');
                                btn.innerHTML = '✓';
                                btn.style.color = '#22c55e';
                            }} catch (e) {{
                                console.error('Fallback copy failed:', e);
                            }}
                            document.body.removeChild(textarea);
                        }});
                    }}
                    </script>
                    """
                    st.components.v1.html(copy_button_html, height=36)
                with col2:
                    like_icon = "❤️" if is_liked else "👍"
                    if st.button(like_icon, key=f"like_{message_key}", use_container_width=True, help="点赞"):
                        if is_liked:
                            st.session_state[f"liked_{message_key}"] = False
                        else:
                            st.session_state[f"liked_{message_key}"] = True
                            st.session_state[f"disliked_{message_key}"] = False
                        message['likes'] = message.get('likes', 0) + (1 if not is_liked else -1)
                        message['dislikes'] = max(0, message.get('dislikes', 0) - (1 if is_disliked else 0))
                        save_session(st.session_state.current_session_id, st.session_state.chat_history)
                        st.rerun()
                with col3:
                    dislike_icon = "💔" if is_disliked else "👎"
                    if st.button(dislike_icon, key=f"dislike_{message_key}", use_container_width=True, help="点踩"):
                        if is_disliked:
                            st.session_state[f"disliked_{message_key}"] = False
                        else:
                            st.session_state[f"disliked_{message_key}"] = True
                            st.session_state[f"liked_{message_key}"] = False
                        message['dislikes'] = message.get('dislikes', 0) + (1 if not is_disliked else -1)
                        message['likes'] = max(0, message.get('likes', 0) - (1 if is_liked else 0))
                        save_session(st.session_state.current_session_id, st.session_state.chat_history)
                        st.rerun()
                with col4:
                    pass


def setup_logging():
    log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler()
        ]
    )


def main():
    setup_logging()
    
    st.set_page_config(
        page_title="智能对话助理智能体",
        page_icon="🤖 ",
        layout="wide"
    )

    st.markdown("""
    <style>
    .user-message {
        background-color: #e8f4fd;
        padding: 12px 16px;
        border-radius: 12px;
        margin-bottom: 8px;
        max-width: 80%;
    }
    .assistant-message {
        background-color: #f5f5f5;
        padding: 12px 16px;
        border-radius: 12px;
        margin-bottom: 8px;
        max-width: 80%;
    }
    .mode-rag {
        background-color: #667eea;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        margin-bottom: 4px;
        display: inline-block;
    }
    .mode-chat {
        background-color: #10b981;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        margin-bottom: 4px;
        display: inline-block;
    }
    .source-card {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 8px;
    }
    .thinking-step {
        color: #666;
        font-style: italic;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = utils_load_uploaded_files(UPLOADED_FILES_LIST)
    if 'last_test_result' not in st.session_state:
        st.session_state.last_test_result = None
    if 'llm_base_url' not in st.session_state:
        st.session_state.llm_base_url = OLLAMA_BASE_URL
    if 'llm_model' not in st.session_state:
        st.session_state.llm_model = OLLAMA_MODEL
    if 'recall_k' not in st.session_state:
        st.session_state.recall_k = 20
    if 'rerank_k' not in st.session_state:
        st.session_state.rerank_k = 5
    if 'use_rerank' not in st.session_state:
        st.session_state.use_rerank = True
    if 'use_web_search' not in st.session_state:
        st.session_state.use_web_search = False
    if 'use_knowledge_search' not in st.session_state:
        st.session_state.use_knowledge_search = False
    if 'search_results_count' not in st.session_state:
        st.session_state.search_results_count = 5
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    if 'show_thinking' not in st.session_state:
        st.session_state.show_thinking = False
    if 'page' not in st.session_state:
        st.session_state.page = 'main'
    if 'kb_manager' not in st.session_state:
        st.session_state.kb_manager = KBManager(Path(__file__).parent)
    
    if 'rag_engine' not in st.session_state:
        from src.core.rag_engine import RAGEngine
        current_kb = st.session_state.get('current_kb', '默认知识库')
        vector_store_path = st.session_state.kb_manager.get_kb_vector_store_path(current_kb)
        st.session_state.rag_engine = RAGEngine(
                            vector_store_path=vector_store_path,
                            embedding_model_name=EMBEDDING_MODEL_NAME,
                            rerank_model_name=RERANK_MODEL_NAME
                        )
        st.session_state.last_test_result = None
        
        def warm_up_llm():
            try:
                def do_warmup():
                    try:
                        llm_client = st.session_state.rag_engine.llm_client
                        for chunk in llm_client.chat_stream("hello"):
                            if chunk.get("done"):
                                break
                    except Exception:
                        pass
                threading.Thread(target=do_warmup, daemon=True).start()
            except Exception:
                pass

    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = generate_session_id()
    
    sessions = load_sessions()
    
    if st.session_state.page == 'kb_manager':
        render_kb_manager_page()
        return
    
    if st.session_state.page == 'doc_preview':
        st.title("📋 文档列表预览")
        if st.button("← 返回主页面", key="back_from_doc"):
            st.session_state.page = 'main'
            st.rerun()
        
        st.markdown("---")
        
        current_kb = st.session_state.get('current_kb', '默认知识库')
        upload_path = st.session_state.kb_manager.get_kb_upload_path(current_kb)
        
        if upload_path.exists():
            documents = [f.name for f in upload_path.iterdir() if f.is_file() and not f.name.startswith('.')]
        else:
            documents = []
        
        st.markdown(f"### 共 {len(documents)} 个文档")
        
        for doc_name in documents:
            st.markdown(f"#### 📄 {doc_name}")
            chunks = st.session_state.rag_engine.vector_store.get_chunks_by_document(doc_name)
            if chunks:
                st.markdown(f"**包含 {len(chunks)} 个片段:**")
                for i, chunk in enumerate(chunks, 1):
                    with st.expander(f"片段 {i}", expanded=False):
                        st.markdown(f"{chunk['content']}")
            else:
                st.markdown("*该文件尚未被索引*")
            st.markdown("---")
        return
    
    if st.session_state.page == 'chunk_preview':
        st.title("📑 片段列表预览")
        if st.button("← 返回主页面", key="back_from_chunk"):
            st.session_state.page = 'main'
            st.rerun()
        
        st.markdown("---")
        chunks = st.session_state.rag_engine.vector_store.get_all_chunks()
        st.markdown(f"### 共 {len(chunks)} 个片段")
        
        for i, chunk in enumerate(chunks, 1):
            st.markdown(f"#### 片段 {i}")
            st.markdown(f"**来源:** {chunk['metadata'].get('source', '未知')}")
            st.markdown("**内容:**")
            st.markdown(chunk['content'])
            st.markdown("---")
        return
    
    if st.session_state.page == 'file_preview':
        st.title(f"📄 文件预览: {st.session_state.preview_file}")
        if st.button("← 返回主页面", key="back_from_file_preview"):
            st.session_state.page = 'main'
            st.rerun()
        
        st.markdown("---")
        
        current_kb = st.session_state.get('current_kb', '默认知识库')
        upload_path = st.session_state.kb_manager.get_kb_upload_path(current_kb)
        file_path = upload_path / st.session_state.preview_file
        
        if not file_path.exists():
            st.error("文件不存在")
            return
        
        kb_available = is_kkfileview_available()
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**文件路径:** `{file_path}`")
        with col2:
            if kb_available:
                st.success("✅ KKFileView 服务已就绪")
            else:
                # 显示Docker状态
                from src.services.kkfileview_service import check_docker_status
                is_installed, is_running, docker_msg = check_docker_status()
                
                if not is_installed:
                    st.error("❌ Docker未安装")
                    st.caption("📥 [下载 Docker Desktop](https://www.docker.com/products/docker-desktop/)")
                elif not is_running:
                    st.warning("⚠️ Docker未运行")
                    
                    # 初始化会话状态
                    if 'waiting_for_docker' not in st.session_state:
                        st.session_state.waiting_for_docker = False
                    
                    # 检查是否在等待Docker启动
                    if st.session_state.waiting_for_docker:
                        # 检查Docker状态
                        from src.services.kkfileview_service import check_docker_status
                        is_installed, is_running_now, _ = check_docker_status()
                        
                        if is_running_now:
                            # Docker已启动，自动启动KKFileView服务
                            st.session_state.waiting_for_docker = False
                            st.success("✅ Docker Desktop 已就绪！")
                            
                            # 自动启动KKFileView服务
                            from src.services.kkfileview_service import start_kkfileview_service
                            progress_container = st.empty()
                            progress_container.info("🚀 正在启动 KKFileView 服务...")
                            
                            def progress_callback(msg):
                                progress_container.info(msg)
                            
                            success, message = start_kkfileview_service(progress_callback=progress_callback)
                            progress_container.empty()
                            
                            if success:
                                st.success(message)
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(message)
                        else:
                            # Docker仍在启动中
                            st.info("⏳ 正在等待 Docker Desktop 启动...")
                            st.caption("💡 请确保 Docker Desktop 正在启动中（可能需要1-2分钟）...")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("🔄 刷新检查", key="refresh_docker_status", use_container_width=True):
                                    st.rerun()
                            with col2:
                                if st.button("取消", key="cancel_waiting"):
                                    st.session_state.waiting_for_docker = False
                                    st.rerun()
                    else:
                        # 初始状态，显示启动按钮
                        if st.button("🐳 启动 Docker Desktop", key="start_docker_btn", use_container_width=True):
                            from src.services.kkfileview_service import start_docker_desktop
                            success, message = start_docker_desktop()
                            
                            if success:
                                st.session_state.waiting_for_docker = True
                                st.rerun()
                            else:
                                st.error(message)
                        
                        st.caption("💡 Docker Desktop启动后，系统将自动检测并启动KKFileView服务")
                else:
                    st.warning("⚠️ KKFileView 服务未启动")
                    
                    # 添加启动按钮
                    if st.button("🚀 启动服务", key="start_kkfileview_btn", use_container_width=True):
                        from src.services.kkfileview_service import start_kkfileview_service
                        
                        # 创建进度提示容器
                        progress_container = st.empty()
                        progress_container.info("🔍 正在检查Docker状态...")
                        
                        def progress_callback(message):
                            progress_container.info(message)
                        
                        success, message = start_kkfileview_service(progress_callback=progress_callback)
                        progress_container.empty()
                        
                        if success:
                            st.success(message)
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(message)
        
        st.markdown("---")
        
        if kb_available:
            try:
                preview_url = get_file_preview_url(str(file_path))
                
                st.markdown("### 文件预览")
                st.markdown(f"""
                <style>
                .preview-container {{
                    position: relative;
                    width: 100%;
                    height: 700px;
                    border-radius: 12px;
                    overflow: hidden;
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
                }}
                </style>
                <div class="preview-container">
                    <iframe src="{preview_url}" width="100%" height="100%" frameborder="0" allowfullscreen></iframe>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                col_download, col_open = st.columns(2)
                with col_download:
                    with open(file_path, 'rb') as f:
                        st.download_button(
                            label="📥 下载文件",
                            data=f,
                            file_name=st.session_state.preview_file,
                            use_container_width=True
                        )
                with col_open:
                    st.markdown(f"""
                    <a href="{preview_url}" target="_blank" style="
                        display: block;
                        width: 100%;
                        padding: 10px 20px;
                        background-color: #667eea;
                        color: white;
                        text-align: center;
                        border-radius: 8px;
                        text-decoration: none;
                        font-weight: 500;
                    ">
                        🔗 在新窗口打开预览
                    </a>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"加载预览时出错: {str(e)}")
                render_text_preview(file_path, "备用文本预览")
        else:
            render_text_preview(file_path, "文本预览（KKFileView未启动）")
        
        return
    
    st.markdown("<h1 style='color: #000000;'>🤖 智能对话助理智能体</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #000000;'>简介： 基于 Streamlit 与 RAG 技术的智能问答系统，支持多轮对话、会话历史管理、多知识库管理、文档上传、文档在线预览与下载、文档向量化存储、知识库检索、联网搜索、思考过程展示等功能。</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    if 'sidebar_collapsed' not in st.session_state:
        st.session_state.sidebar_collapsed = False
    
    if st.session_state.sidebar_collapsed:
        main_content_ratio = 0.75
        input_align_left = True
    else:
        main_content_ratio = 0.6
        input_align_left = False
    
    if input_align_left:
        st.markdown(f"""
        <style>
        .stChatInput {{
            max-width: {main_content_ratio * 100}% !important;
            margin-right: auto !important;
            margin-left: 0 !important;
        }}
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <style>
        .stChatInput {{
            max-width: {main_content_ratio * 100}% !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }}
        </style>
        """, unsafe_allow_html=True)
    
    if st.session_state.sidebar_collapsed:
        col_main, col_kb = st.columns([3, 1])
        
        with col_kb:
            with st.container(border=True):
                render_knowledge_base_section()
        
        with col_main:
            with st.container(border=True):
                if st.button("📋 展开会话历史", use_container_width=False):
                    st.session_state.sidebar_collapsed = False
                    st.rerun()
                
                render_llm_config_section()
                
                st.markdown("---")
                
                st.markdown("### 🚀 推理选项")
                col_search1, col_search2, col_search3 = st.columns([1, 1, 1])
                with col_search1:
                    use_knowledge_search = st.toggle("📚 知识库搜索", value=st.session_state.use_knowledge_search, key="knowledge_search_toggle")
                    st.session_state.use_knowledge_search = use_knowledge_search
                with col_search2:
                    use_web_search = st.toggle("🌐 联网搜索", value=st.session_state.use_web_search, key="web_search_toggle")
                    st.session_state.use_web_search = use_web_search
                with col_search3:
                    show_thinking = st.toggle("🧠 思考过程", value=st.session_state.show_thinking, key="thinking_toggle")
                    st.session_state.show_thinking = show_thinking
                
                st.markdown("💡 提示：启用知识库搜索将从上传的文档中检索相关信息，启用联网搜索将从互联网获取实时信息。")
                
                st.markdown("---")
                
                st.markdown("### 💬 对话")
                render_chat_interface()
    else:
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col1:
            with st.container(border=True):
                col_title, col_collapse = st.columns([4, 1])
                with col_title:
                    st.markdown("### 📋 会话历史")
                with col_collapse:
                    if st.button("«", key="collapse_sidebar", help="隐藏会话历史"):
                        st.session_state.sidebar_collapsed = True
                        st.rerun()
                
                if st.button("➕ 新建会话", use_container_width=True):
                    st.session_state.current_session_id = generate_session_id()
                    st.session_state.chat_history = []
                    if 'rag_engine' in st.session_state and st.session_state.rag_engine._llm_client:
                        st.session_state.rag_engine._llm_client.clear_history()
                    st.rerun()
                
                st.markdown("---")
                
                for session in sessions:
                    is_active = session['id'] == st.session_state.current_session_id
                    col_btn, col_del = st.columns([5, 1])
                    with col_btn:
                        if st.button(
                            session['title'],
                            key=f"session_{session['id']}",
                            use_container_width=True,
                            type="primary" if is_active else "secondary"
                        ):
                            st.session_state.current_session_id = session['id']
                            session_data = load_session(session['id'])
                            if session_data:
                                st.session_state.chat_history = session_data.get('chat_history', [])
                            if 'rag_engine' in st.session_state and st.session_state.rag_engine._llm_client:
                                st.session_state.rag_engine._llm_client.clear_history()
                            st.rerun()
                    with col_del:
                        if st.button("x", key=f"del_{session['id']}", use_container_width=True):
                            delete_session(session['id'])
                            if session['id'] == st.session_state.current_session_id:
                                st.session_state.current_session_id = generate_session_id()
                                st.session_state.chat_history = []
                            st.rerun()
        
        with col3:
            with st.container(border=True):
                render_knowledge_base_section()
        
        with col2:
            with st.container(border=True):
                render_llm_config_section()
                
                st.markdown("---")
                
                st.markdown("### 🚀 推理选项")
                col_search1, col_search2, col_search3 = st.columns([1, 1, 1])
                with col_search1:
                    use_knowledge_search = st.toggle("📚 知识库搜索", value=st.session_state.use_knowledge_search, key="knowledge_search_toggle")
                    st.session_state.use_knowledge_search = use_knowledge_search
                with col_search2:
                    use_web_search = st.toggle("🌐 联网搜索", value=st.session_state.use_web_search, key="web_search_toggle")
                    st.session_state.use_web_search = use_web_search
                with col_search3:
                    show_thinking = st.toggle("🧠 思考过程", value=st.session_state.show_thinking, key="thinking_toggle")
                    st.session_state.show_thinking = show_thinking
                
                st.markdown("💡 提示：启用知识库搜索将从上传的文档中检索相关信息，启用联网搜索将从互联网获取实时信息。")
                
                st.markdown("---")
                
                st.markdown("### 💬 对话")
                render_chat_interface()
    
    user_input = st.chat_input(
        "请输入您的问题，我会根据知识库为您解答",
        key="user_input"
    )
    
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        
        full_response = ""
        current_mode = "chat"
        sources = []
        search_results = []
        
        with st.chat_message("assistant"):
            think_placeholder = None
            if st.session_state.show_thinking:
                with st.expander("🧠 思考过程（实时）", expanded=True):
                    think_placeholder = st.empty()
            
            answer_box = st.empty()
            
            think_buf = ""
            answer_buf = ""
            message_idx = len(st.session_state.chat_history) + 1
            
            stats = st.session_state.rag_engine.vector_store.get_stats()
            has_knowledge_base = stats['total_documents'] > 0
            use_web_search = st.session_state.use_web_search
            use_knowledge_search = st.session_state.use_knowledge_search
            
            contexts = []
            retrieved_docs = None
            search_results = []
            use_rag = has_knowledge_base and use_knowledge_search
            
            feedback_history = []
            for i, msg in enumerate(st.session_state.chat_history):
                if msg.get('role') == 'assistant' and i > 0:
                    user_msg = st.session_state.chat_history[i - 1]
                    if user_msg.get('role') == 'user':
                        likes = msg.get('likes', 0)
                        dislikes = msg.get('dislikes', 0)
                        rating = 0
                        if likes > dislikes:
                            rating = 1
                        elif dislikes > likes:
                            rating = -1
                        if rating != 0:
                            feedback_history.append({
                                'query': user_msg.get('content', ''),
                                'response': msg.get('content', ''),
                                'rating': rating
                            })
            
            if feedback_history:
                display_feedback = feedback_history[-5:]
                with st.expander(f"🔍 当前反馈历史 ({len(display_feedback)} 条，最多显示5条)", expanded=False):
                    for idx, feedback in enumerate(display_feedback):
                        rating_icon = "❤️" if feedback['rating'] > 0 else "💔"
                        rating_text = "好评" if feedback['rating'] > 0 else "改进"
                        st.markdown(f"**{rating_icon} {rating_text} {idx + 1}:**")
                        st.markdown(f"**用户问:** {feedback['query'][: 50]}...")
                        st.markdown(f"**回答:** {feedback['response'][: 100]}...")
                        st.divider()
            
            result_queue = queue.Queue()
            
            def retrieve_docs():
                try:
                    retrieved_docs = st.session_state.rag_engine.retriever.retrieve(
                        query=user_input,
                        recall_top_k=st.session_state.recall_k,
                        rerank_top_k=st.session_state.rerank_k if st.session_state.use_rerank else st.session_state.recall_k
                    )
                    result_queue.put(('retrieve', retrieved_docs))
                except Exception as e:
                    result_queue.put(('retrieve_error', str(e)))
            
            def web_search():
                try:
                    searcher = WebSearch()
                    num_results = st.session_state.get('search_results_count', 5)
                    if num_results > 0:
                        results = searcher.search_with_content(user_input, num_results=num_results)
                        result_queue.put(('search', results))
                    else:
                        result_queue.put(('search', []))
                except Exception as e:
                    result_queue.put(('search_error', str(e)))
            
            if use_rag:
                if st.session_state.show_thinking:
                    think_buf = "🔍 正在检索知识库..."
                    think_placeholder.markdown(think_buf)
                try:
                    retrieved_docs = st.session_state.rag_engine.retriever.retrieve(
                        query=user_input,
                        recall_top_k=st.session_state.recall_k,
                        rerank_top_k=st.session_state.rerank_k if st.session_state.use_rerank else st.session_state.recall_k
                    )
                    if st.session_state.show_thinking:
                        think_buf += f"\n✅ 知识库检索完成，获取到{len(retrieved_docs)}个相关片段"
                        think_placeholder.markdown(think_buf)
                    sources.extend([
                        {
                            'content': doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'],
                            'source': doc['metadata']['source'],
                            'score': doc.get('rerank_score', doc.get('distance', 0)),
                            'type': 'knowledge'
                        }
                        for doc in retrieved_docs
                    ])
                    contexts.extend([doc['content'] for doc in retrieved_docs])
                except Exception as e:
                    if st.session_state.show_thinking:
                        think_buf += f"\n⚠️ 知识库检索失败: {str(e)}"
                        think_placeholder.markdown(think_buf)
            
            if use_web_search:
                if st.session_state.show_thinking:
                    if think_buf:
                        think_buf += "\n"
                    think_buf += "🌐 正在进行联网搜索..."
                    think_placeholder.markdown(think_buf)
                try:
                    searcher = WebSearch()
                    num_results = st.session_state.get('search_results_count', 5)
                    if num_results > 0:
                        search_results = searcher.search_with_content(user_input, num_results=num_results)
                    else:
                        search_results = []
                    
                    if st.session_state.show_thinking:
                        think_buf += f"\n✅ 联网搜索完成，获取到{len(search_results)}条结果：\n"
                        for r in search_results:
                            think_buf += f"- [{r['title']}]({r['url']})\n"
                        think_placeholder.markdown(think_buf)
                    sources.extend([
                        {
                            'content': r['content'][:200] + "..." if len(r['content']) > 200 else r['content'],
                            'source': r['title'],
                            'url': r['url'],
                            'score': 1.0 - (r['rank'] * 0.1),
                            'type': 'web'
                        }
                        for r in search_results
                    ])
                    contexts.extend([f"【{r['title']}】\n{r['content']}" for r in search_results])
                except Exception as e:
                    if st.session_state.show_thinking:
                        think_buf += f"\n⚠️ 联网搜索失败: {str(e)}"
                        think_placeholder.markdown(think_buf)
            
            if st.session_state.show_thinking:
                if think_buf:
                    think_buf += "\n\n"
                think_buf += "🧠 正在分析并生成答案..."
                think_placeholder.markdown(think_buf)
            
            if contexts:
                for chunk in st.session_state.rag_engine.llm_client.generate_with_context_stream(
                    query=user_input, 
                    context=contexts,
                    think=st.session_state.show_thinking,
                    feedback_history=feedback_history
                ):
                    if chunk.get("thinking") and st.session_state.show_thinking:
                        think_buf += chunk["thinking"]
                        think_placeholder.markdown(think_buf)
                    if chunk.get("content"):
                        answer_buf += chunk["content"]
                        answer_box.markdown(answer_buf)
            else:
                for chunk in st.session_state.rag_engine.llm_client.chat_stream(
                    query=user_input,
                    think=st.session_state.show_thinking,
                    feedback_history=feedback_history
                ):
                    if chunk.get("thinking") and st.session_state.show_thinking:
                        think_buf += chunk["thinking"]
                        think_placeholder.markdown(think_buf)
                    if chunk.get("content"):
                        answer_buf += chunk["content"]
                        answer_box.markdown(answer_buf)
        
        full_response = answer_buf
        
        message_key = f"msg_{message_idx}"
        is_liked = st.session_state.get(f"liked_{message_key}", False)
        is_disliked = st.session_state.get(f"disliked_{message_key}", False)
        copied = st.session_state.get(f"copied_{message_key}", False)
        
        col1, col2, col3, col4 = st.columns([0.8, 0.8, 0.8, 5])
        with col1:
            copy_icon = "✓" if copied else "📋"
            copy_color = "#22c55e" if copied else "#6b7280"
            copy_button_html = f"""
            <button 
                id="copy-btn-{message_key}"
                onclick="copyToClipboard_{message_key}()"
                style="
                    width: 100%;
                    padding: 6px 12px;
                    border: none;
                    border-radius: 8px;
                    cursor: pointer;
                    font-size: 18px;
                    background-color: transparent;
                    color: {copy_color};
                    transition: all 0.2s;
                "
                onmouseover="this.style.opacity='0.7'"
                onmouseout="this.style.opacity='1'"
                title="复制回答"
            >
                {copy_icon}
            </button>
            <script>
            function copyToClipboard_{message_key}() {{
                const text = {json.dumps(full_response)};
                navigator.clipboard.writeText(text).then(function() {{
                    const btn = document.getElementById('copy-btn-{message_key}');
                    btn.innerHTML = '✓';
                    btn.style.color = '#22c55e';
                }}).catch(function(err) {{
                    console.error('Failed to copy:', err);
                    const textarea = document.createElement('textarea');
                    textarea.value = text;
                    document.body.appendChild(textarea);
                    textarea.select();
                    try {{
                        document.execCommand('copy');
                        const btn = document.getElementById('copy-btn-{message_key}');
                        btn.innerHTML = '✓';
                        btn.style.color = '#22c55e';
                    }} catch (e) {{
                        console.error('Fallback copy failed:', e);
                    }}
                    document.body.removeChild(textarea);
                }});
            }}
            </script>
            """
            st.components.v1.html(copy_button_html, height=36)
        with col2:
            like_icon = "❤️" if is_liked else "👍"
            if st.button(like_icon, key=f"like_{message_key}", use_container_width=True, help="点赞"):
                if is_liked:
                    st.session_state[f"liked_{message_key}"] = False
                else:
                    st.session_state[f"liked_{message_key}"] = True
                    st.session_state[f"disliked_{message_key}"] = False
                st.rerun()
        with col3:
            dislike_icon = "💔" if is_disliked else "👎"
            if st.button(dislike_icon, key=f"dislike_{message_key}", use_container_width=True, help="点踩"):
                if is_disliked:
                    st.session_state[f"disliked_{message_key}"] = False
                else:
                    st.session_state[f"disliked_{message_key}"] = True
                    st.session_state[f"liked_{message_key}"] = False
                st.rerun()
        with col4:
            pass
        
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input
        })
        
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': full_response,
            'thinking': think_buf,
            'mode': current_mode,
            'sources': sources,
            'search_results': search_results,
            'likes': st.session_state.get(f"like_{message_key}", 0),
            'dislikes': st.session_state.get(f"dislike_{message_key}", 0)
        })
        
        save_session(st.session_state.current_session_id, st.session_state.chat_history)


if __name__ == '__main__':
    main()