#!/bin/bash
# scripts/run.sh

echo "🚀 启动 RTX3090 RAG 知识库服务..."

# 确保上传目录存在
mkdir -p uploaded_docs

# 启动 Gradio 应用
python src/rag_app.py

echo "✅ 服务已启动，请通过 AutoDL 实例链接访问！"