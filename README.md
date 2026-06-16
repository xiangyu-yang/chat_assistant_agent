# 智能对话助理智能体（RAG-based Chat Assistant Agent）

基于 Streamlit 和 RAG 技术的智能问答系统，支持多知识库管理、文档上传、向量化存储、知识库检索、分片策略配置和联网搜索等功能。

## 技术栈

- **Streamlit**: Web UI 框架
- **FAISS**: 高效向量相似度搜索
- **Sentence Transformers**: 文本嵌入和重排模型
- **Ollama**: 本地大模型服务
- **BAAI/bge-m3**: 嵌入模型（多语言、高性能）
- **BAAI/bge-reranker-v2-m3**: 重排模型（提升检索精度）
- **Hugging Face**: 模型下载（国内通过 hf-mirror 加速）
- **KKFileView**: 在线文件预览服务（Docker）
- **Docker Desktop**: 容器化部署和管理

## 核心功能

### 🤖 智能问答
- 基于本地大模型的对话功能
- 支持多轮对话上下文理解
- 实时流式响应
- 思考过程展示
- 参考来源展示

### 📚 知识库管理
- **多知识库支持**: 创建和管理多个独立知识库
- **文档上传**: 支持 PDF、DOCX、TXT、Excel 格式
- **智能检索**: Recall + ReRank 两阶段检索
- **向量存储**: FAISS 高性能向量索引
- **文档预览**: 在线预览上传的文档（KKFileView）
- **片段预览**: 查看向量化后的文档片段
- **一键启动**: 自动检测并启动 Docker Desktop 和 KKFileView 服务

### ⚙️ 分片策略配置
支持三种文档分片策略：
- **📏 固定长度分片**: 按固定字符数分割，保持简单高效，适合结构化文档
- **📊 父子层次分片**: 先按章节标题分割，再在章节内分片，适合有明确章节结构的文档
- **🧠 语义向量分片**: 按句子边界分割，保持语义完整性，适合非结构化文本

可配置参数：
- 分片大小（字符数）
- 重叠大小（相邻分片间的重叠字符数）
- 召回数量、重排数量

### 🌐 联网搜索
- 集成网络搜索功能
- 支持配置搜索结果数量
- 搜索结果与知识库结果混合展示

### 💬 会话管理
- 新建和切换会话
- 会话历史自动保存
- 删除会话功能
- 会话侧栏可折叠

### ⚙️ LLM 配置
- 动态配置 Ollama 服务地址
- 支持选择可用模型
- 连接测试功能

### 📊 用户反馈
- 回答点赞/点踩功能
- 回答一键复制
- 反馈历史查看

## 快速启动

### 方式一：使用启动脚本（推荐）

```bash
cd mini_assistant
chmod +x startup.sh
./startup.sh
```

### 方式二：手动启动

1. **安装依赖**

```bash
cd mini_assistant
pip install -r requirements.txt
```

2. **启动 Ollama 服务**

```bash
# 启动 Ollama 服务
ollama serve

# 或直接运行模型（首次运行会自动下载）
ollama run qwen3.6:35b-a3b-q8_0
```

3. **启动应用**

```bash
streamlit run app.py --server.port 8501
```

应用将在浏览器中打开，默认地址：http://localhost:8501

## KKFileView 文档预览服务

### 自动启动（推荐）

应用内置了智能检测和自动启动功能：
1. 当您尝试预览文档时，如果 Docker 未运行，系统会显示"🐳 启动 Docker Desktop"按钮
2. 点击按钮即可自动启动 Docker Desktop
3. Docker 启动后，系统会自动检测并启动 KKFileView 服务
4. 整个过程无需手动操作

### 手动启动

如需手动启动 KKFileView 服务：

```bash
# 启动 Docker Desktop
open -a Docker

# 等待 Docker 完全启动后，运行 KKFileView 容器
docker run -d \
  --name kkfileview \
  -p 8012:8012 \
  -v $(pwd)/knowledge_bases:/opt/kkfileview/knowledge_bases \
  --restart=always \
  keenq/kkfileview:4.1.0
```

KKFileView 服务将在 http://localhost:8012 运行。

## 使用流程

### 1. 配置 LLM
- 在 "LLM 配置" 区域设置 Ollama 服务地址和模型
- 点击 "测试连接" 验证配置
- 点击 "应用配置" 保存设置

### 2. 创建和管理知识库
- 点击 "管理知识库" 进入知识库管理页面
- 点击 "新建知识库" 创建新的知识库
- 选择分片策略并配置参数
- 点击 "使用此知识库" 切换到该知识库

### 3. 上传和处理文档
- 在侧边栏选择知识库
- 上传需要建立知识库的文档（支持多文件）
- 选择分片策略（可选，默认固定长度）
- 点击 "处理文档" 等待向量化完成

### 4. 开始问答
- 配置推理选项：启用知识库搜索、启用联网搜索、显示思考过程
- 在输入框中输入问题
- 系统会根据配置进行回答
- 可以查看回答的参考来源、点赞/点踩回答、复制回答

### 5. 预览文档
- 在知识库中找到要预览的文件
- 点击文件进行预览
- 如 KKFileView 未启动，系统会提示并可一键启动

## 配置说明

在 `src/config/settings.py` 中可以修改以下配置，也可以通过环境变量设置：

| 配置项 | 环境变量 | 默认值 | 说明 |
|--------|----------|--------|------|
| `OLLAMA_BASE_URL` | `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama 服务地址 |
| `OLLAMA_MODEL` | `OLLAMA_MODEL` | `qwen3.6:35b-a3b-q8_0` | 大模型名称 |
| `EMBEDDING_MODEL_NAME` | `EMBEDDING_MODEL_NAME` | `BAAI/bge-m3` | 嵌入模型名称 |
| `RERANK_MODEL_NAME` | `RERANK_MODEL_NAME` | `BAAI/bge-reranker-v2-m3` | 重排模型名称 |
| `CHUNK_SIZE` | `CHUNK_SIZE` | `500` | 文档分块大小（字符） |
| `CHUNK_OVERLAP` | `CHUNK_OVERLAP` | `50` | 分块重叠大小（字符） |
| `RECALL_TOP_K` | `RECALL_TOP_K` | `20` | 召回阶段返回的文档数量 |
| `RERANK_TOP_K` | `RERANK_TOP_K` | `5` | 重排后返回的文档数量 |
| `KNOWLEDGE_BASES_DIR` | `KNOWLEDGE_BASES_DIR` | `./knowledge_bases` | 知识库存储目录 |

## Streamlit 配置

在 `.streamlit/config.toml` 中可以配置服务端口等参数：

```toml
[server]
port = 8501
address = "0.0.0.0"
maxUploadSize = 200

[browser]
serverAddress = "localhost"
serverPort = 8501
```

## 项目结构

```
mini_assistant/
├── app.py                            # Streamlit 主应用
├── startup.sh                        # 一键启动脚本
├── requirements.txt                  # 依赖列表
├── README.md                         # 项目文档
├── app.log                           # 应用日志
├── .streamlit/
│   └── config.toml                   # Streamlit 配置
└── src/
    ├── __init__.py
    ├── config/
    │   ├── __init__.py
    │   └── settings.py               # 应用配置文件
    ├── core/
    │   ├── __init__.py
    │   ├── kb_manager.py             # 知识库管理器
    │   └── rag_engine.py             # RAG 引擎整合
    ├── data/
    │   ├── __init__.py
    │   ├── document_processor.py     # 文档处理模块（支持多种分片策略）
    │   └── vector_store.py           # FAISS 向量存储
    ├── models/
    │   ├── __init__.py
    │   ├── embeddings.py             # BGE-M3 嵌入和重排模型
    │   ├── retriever.py              # 检索器（Recall + Rerank）
    │   └── llm.py                    # Ollama LLM 客户端
    ├── services/
    │   ├── __init__.py
    │   ├── search.py                 # 网络搜索服务
    │   └── kkfileview_service.py    # KKFileView 服务管理（自动启动 Docker）
    └── utils/
        ├── __init__.py
        ├── file_utils.py             # 文件工具
        └── session_manager.py        # 会话管理
```

## RAG 流程说明

1. **文档上传**: 支持多格式文档（PDF、DOCX、TXT、Excel）
2. **分片处理**: 根据选择的策略将文档分割成块
   - 固定长度：按字符数均匀分割
   - 父子层次：先按章节分割，再细分
   - 语义向量：按句子边界分割
3. **向量化**: 使用 BGE-M3 将文本块转为向量
4. **存储**: 使用 FAISS 建立向量索引并持久化
5. **Recall**: 根据用户问题，从向量库中召回最相关的文档块
6. **ReRank**: 使用 BGE-Reranker-v2-m3 对召回结果进行重排
7. **联网搜索**（可选）: 从互联网获取实时信息
8. **生成**: 将相关文档和搜索结果作为上下文，调用 LLM 生成回答

## 模型说明

### 嵌入模型: BAAI/bge-m3
- 高性能多语言嵌入模型
- 支持 100+ 种语言
- 在多种检索任务上表现优异

### 重排模型: BAAI/bge-reranker-v2-m3
- 专门优化的重排模型
- 提升检索精度，过滤噪声
- 支持长文本处理

## 注意事项

- 确保 Ollama 服务正在运行
- 首次运行会下载 BGE-M3 和 BGE-Reranker 模型
- 每个知识库独立存储文档和向量索引
- 支持多文件同时上传
- 会话数据自动保存在 `sessions` 目录
- 默认服务端口为 8501
- 文档预览需要 Docker Desktop 运行中
- 系统会自动检测并提示启动所需服务
