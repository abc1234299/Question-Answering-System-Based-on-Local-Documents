# Question-Answering-System-Based-on-Local-Documents
# 📚 基于本地文档的 RAG 问答系统（RTX3090 加速版）

一个基于本地文档的检索增强生成（RAG）问答系统，利用 RTX3090 GPU 加速，支持 PDF/TXT 文档上传、智能问答与引用溯源。

---

## ✨ 功能特点

- 🚀 **GPU 加速**：专为 RTX3090 优化，使用 `m3e-large` 中文嵌入模型和 `qwen2.5:7b` LLM 模型
- 📄 **多格式支持**：支持 PDF 和 TXT 格式文档上传与解析
- 🔍 **智能检索**：基于 FAISS 向量数据库实现语义检索
- 📖 **引用溯源**：回答附带原文引用，支持页码和内容片段展示
- 🌐 **Web 界面**：基于 Gradio 构建友好的交互界面
- 📂 **文档管理**：支持批量上传、清空知识库等操作

---

## 📁 项目结构

| 目录/文件 | 说明 |
|----------|------|
| **docs/** | 默认文档存放目录（存放初始PDF文档） |
| **uploaded_docs/** | 用户上传文档保存目录 |
| **faiss_gpu_db/** | FAISS向量数据库（运行后自动生成） |
| **models/** | 模型文件目录（如需要本地存储模型） |
| **results/** | 运行结果和输出文件 |
| **scripts/** | 辅助脚本和工具 |
| **src/** | 源代码目录（包含主程序） |
| **LICENSE** | 项目许可证文件 |
| **README.md** | 项目说明文档（本文档） |
| **requirement.txt** | Python依赖包列表 |
| **ollama.log** | Ollama模型服务日志 |

## 🔧 安装步骤
```markdown
### 1. 克隆项目
```
git clone https://github.com/abc1234299/Question-Answering-System-Based-on-Local-Documents.git
cd Question-Answering-System-Based-on-Local-Documents

### 2. 安装 Python 依赖
```markdown
pip install langchain langchain_text_splitters faiss-gpu sentence-transformers ollama pypdf gradio -i https://pypi.tuna.tsinghua.edu.cn/simple

### 3. 验证安装
python -c "import langchain; print('✅ LangChain 安装成功')"

### 4. 安装 Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen2.5:7b
```

## 🚀 快速使用
```markdown
### 启动系统
```
python src/main.py

### 访问 Web 界面
http://<服务器IP>:7860


## ⚡ 性能优化配置
针对 RTX3090 显卡特性，以下配置可最大化利用硬件性能，兼顾问答速度和准确率：

### 1. GPU 全链路加速设置
```python
# 嵌入模型 GPU 加速（m3e-large 中文模型）
embeddings = SentenceTransformerEmbeddings(
    model_name="moka-ai/m3e-large",
    model_kwargs={"device": "cuda"}  # 强制使用 GPU 进行嵌入计算
)

# LLM 模型 GPU 分配（qwen2.5:7b）
llm = Ollama(model="qwen2.5:7b", num_gpu=1)  # num_gpu=1 为模型分配全部 GPU 资源
```
### 2. 显存优化
# 方案1：调整文本分割参数（减少单次计算显存占用）
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,  # 适度增大 chunk 尺寸，减少总片段数
    chunk_overlap=150,  # 降低重叠率，减少重复计算
    separators=["\n\n", "\n", "。", "！", "？", "；", "，"]
)
```
# 方案2：LLM 显存限制（避免显存溢出）
```python
llm = Ollama(
    model="qwen2.5:7b",
    num_gpu=1,
    num_ctx=8192,  # 上下文窗口大小（平衡显存和问答能力）
    temperature=0.1  # 降低随机性，减少计算量
)
```
# 方案3：FAISS GPU 索引优化
```python
db = FAISS.from_documents(split_docs, embeddings)
db = db.to_gpu()  # 强制将向量库加载到 GPU，检索速度提升 5-10 倍
```

## 🐛 常见问题
### Q1：依赖安装失败
**问题现象**：执行 `pip install` 时出现包冲突、编译失败或下载超时  
**解决方案**：
```bash
# 方案1：使用虚拟环境隔离依赖（推荐）
python -m venv rag_venv
source rag_venv/bin/activate  # Linux/Mac
# rag_venv\Scripts\activate  # Windows

# 方案2：分步安装，优先解决 faiss-gpu 依赖问题
pip install faiss-gpu==1.7.2 --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install langchain sentence-transformers pypdf gradio ollama -i https://pypi.tuna.tsinghua.edu.cn/simple

# 方案3：降级 pip 版本（适配部分系统）
pip install pip==23.0.1
```
### Q2：Ollama 模型下载慢 / 失败
**问题现象**：ollama pull qwen2.5:7b 速度极慢或提示连接超时解决方案：
**解决方案**
```bash
# 临时配置镜像源（单次生效）
export OLLAMA_HOST=https://mirror.ghproxy.com
ollama pull qwen2.5:7b

# 永久配置镜像源（Linux）
echo 'export OLLAMA_HOST=https://mirror.ghproxy.com' >> ~/.bashrc
source ~/.bashrc
ollama pull qwen2.5:7b

# 备选方案：手动下载模型文件后导入
# 1. 下载模型文件到本地
# 2. 执行：ollama create qwen2.5:7b -f ./Modelfile
```
### Q3：GPU 内存不足（OOM 报错
**问题现象**：运行时提示 CUDA out of memory 或程序崩溃解决方案：
**解决方案**
```python
# 方案1：减少检索片段数量（降低 LLM 推理压力）
retriever = db.as_retriever(search_kwargs={"k": 2})  # 从 4 降至 2

# 方案2：更换轻量嵌入模型 + 调整批处理大小
embeddings = SentenceTransformerEmbeddings(
    model_name="moka-ai/m3e-base",  # 替换为基础版（显存占用减少 50%）
    model_kwargs={"device": "cuda", "batch_size": 16}  # 降低批处理大小
)

# 方案3：使用更小的 LLM 模型
llm = Ollama(model="qwen2.5:4b", num_gpu=1)  # 4B 模型替代 7B 模型

# 方案4：启用显存分片（终极方案）
import torch
torch.cuda.empty_cache()  # 清理显存缓存
embeddings = SentenceTransformerEmbeddings(
    model_name="moka-ai/m3e-large",
    model_kwargs={"device": "cuda", "trust_remote_code": True, "load_in_8bit": True}
)
```
