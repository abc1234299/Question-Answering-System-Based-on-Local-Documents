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

### 1. 克隆项目
```bash
git clone <项目地址>
cd Question-Answering-System-Based-on-Local-Documents


## 📦 步骤 2：安装依赖

### 安装核心依赖包
```bash
# 使用清华镜像源加速安装
pip install langchain langchain_text_splitters faiss-gpu sentence-transformers ollama pypdf gradio -i https://pypi.tuna.tsinghua.edu.cn/simple

### 2. 安装 Python 依赖

```bash
pip install langchain langchain_text_splitters faiss-gpu sentence-transformers ollama pypdf gradio -i https://pypi.tuna.tsinghua.edu.cn/simple