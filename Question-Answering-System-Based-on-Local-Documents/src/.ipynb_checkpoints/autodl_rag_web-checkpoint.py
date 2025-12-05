# 依赖安装命令（终端先执行）
# pip install langchain langchain_text_splitters faiss-gpu sentence-transformers ollama pypdf gradio -i https://pypi.tuna.tsinghua.edu.cn/simple

from langchain_text_splitters import RecursiveCharacterTextSplitter
# 保留其他必要导入，只修改这两行关于嵌入和向量存储的
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings  # 关键：导入正确的类
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import gradio as gr
import os
import shutil

# RTX3090专属配置
EMBED_MODEL = "m3e-large"  # 中文GPU嵌入模型（精度高、速度快）
LLM_MODEL = "qwen2.5:7b"  # 7B模型（RTX3090轻松承载）
VECTOR_DB_PATH = "faiss_gpu_db"
UPLOAD_DIR = "uploaded_docs"  # 上传文档保存目录


# 初始化GPU组件
embeddings = SentenceTransformerEmbeddings(
    model_name="moka-ai/m3e-large",  # 让程序自动从国内源下载
    model_kwargs={"device": "cuda"}
)
llm = Ollama(model=LLM_MODEL, num_gpu=1)  # 分配GPU给LLM

def build_or_load_db():
    # 1. 导入所需依赖（确保不缺模块）
    from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    global embeddings  # 引用全局的 embeddings 变量（你已在函数外初始化）

    # 2. 配置文档加载器（加载 docs 文件夹下所有 PDF）
    loader = DirectoryLoader(
        path="/root/autodl-tmp/docs",  # 文档文件夹绝对路径（和你创建的一致）
        glob="*.pdf",  # 仅加载 PDF 文件，避免无关文件干扰
        loader_cls=PyPDFLoader,  # 指定 PDF 解析器
        show_progress=True  # 显示加载进度（可选，方便查看加载状态）
    )

    # 3. 加载文档并验证
    try:
        docs = loader.load()
        print(f"\n✅ 成功从 docs 文件夹加载到 {len(docs)} 个 PDF 文档")
    except Exception as e:
        print(f"\n❌ 文档加载失败：{str(e)}")
        print("请检查：1. docs 文件夹是否存在 2. 文件夹内是否有 PDF 文件 3. 已安装 pypdf 依赖（pip install pypdf）")
        return None

    # 4. 初始化文本分割器（解决 NameError，适配长文档）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # 每个片段的字符数（可调整，如 1500 适合长文本）
        chunk_overlap=200,  # 片段间重叠字符数（避免割裂语义，建议为 chunk_size 的 10%-20%）
        length_function=len,  # 按字符长度计算（中文友好）
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]  # 中文优先分割符
    )

    # 5. 分割文档并验证
    split_docs = text_splitter.split_documents(docs)
    print(f"✅ 文档分割完成，得到 {len(split_docs)} 个文档片段")

    # 6. 构建并保存 FAISS 向量库
    if len(split_docs) > 0:
        try:
            db = FAISS.from_documents(split_docs, embeddings)
            db.save_local("faiss_db")  # 保存到本地，下次可直接加载
            print(f"✅ FAISS 向量库创建成功！已保存到 faiss_db 文件夹")
            return db
        except Exception as e:
            print(f"\n❌ 向量库创建失败：{str(e)}")
            return None
    else:
        print(f"\n❌ 没有可用的文档片段（分割后数量为 0），请检查 PDF 内容是否为空或调整分割参数")
        return None
# 上传文档并更新向量库
def upload_docs(files):
    if not files:
        return "未上传任何文档！", None
    # 保存上传的文件
    for file in files:
        shutil.copy(file, os.path.join(UPLOAD_DIR, os.path.basename(file)))
    # 重建向量库
    db = build_or_load_db()
    return f"✅ 成功上传 {len(files)} 个文档，已更新知识库！", db

# RAG问答核心函数
def rag_qa(query, db):
    if not db:
        return "❌ 知识库为空，请先上传文档！", ""
    retriever = db.as_retriever(search_kwargs={"k": 4})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",
        retriever=retriever,
        return_source_documents=True
    )
    result = qa_chain({"query": query})
    
    # 整理引用来源
    sources = ""
    for i, doc in enumerate(result["source_documents"][:3], 1):
        filename = os.path.basename(doc.metadata["source"])
        page = doc.metadata.get("page", 0) + 1  # PDF页码从1开始
        content = doc.page_content[:180] + "..." if len(doc.page_content) > 180 else doc.page_content
        sources += f"\n【引用{i}】《{filename}》第{page}页：{content}"
    
    return result["result"], f"🔍 参考来源：{sources}"

# 构建Gradio Web界面
with gr.Blocks(title="RTX3090 RAG知识库") as demo:
    gr.Markdown("# 📚 本地知识库问答（RTX3090加速）")
    gr.Markdown("支持上传PDF/TXT文档，基于文档内容精准问答（带引用溯源）")
    
    # 存储向量库实例（全局变量）
    db_state = gr.State(None)
    
    with gr.Row():
        with gr.Column(scale=1):
            file_upload = gr.File(
                label="上传文档（支持多文件）",
                file_types=[".pdf", ".txt"],
                file_count="multiple"
            )
            upload_btn = gr.Button("📤 上传并更新知识库")
            upload_status = gr.Textbox(label="上传状态", interactive=False)
            clear_btn = gr.Button("🗑️ 清空知识库")
        
        with gr.Column(scale=2):
            query_input = gr.Textbox(label="请提问", placeholder="例如：文档中提到的核心结论是什么？")
            qa_btn = gr.Button("🚀 开始问答")
            answer_output = gr.Textbox(label="回答", lines=8, interactive=False)
            source_output = gr.Textbox(label="引用来源", lines=4, interactive=False)
    
    # 绑定按钮事件
    upload_btn.click(
        fn=upload_docs,
        inputs=file_upload,
        outputs=[upload_status, db_state]
    )
    
    qa_btn.click(
        fn=rag_qa,
        inputs=[query_input, db_state],
        outputs=[answer_output, source_output]
    )
    
    # 清空知识库
    def clear_knowledge():
        if os.path.exists(VECTOR_DB_PATH):
            shutil.rmtree(VECTOR_DB_PATH)
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
            os.makedirs(UPLOAD_DIR)
        return "✅ 知识库已清空！", None
    clear_btn.click(fn=clear_knowledge, outputs=[upload_status, db_state])

# 启动Web服务（适配AutoDL端口）
if __name__ == "__main__":
    # 初始化向量库（首次运行为空）
    build_or_load_db()
    demo.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860,        # AutoDL默认自定义端口
        show_error=True,
        share=False  # AutoDL无需额外分享，直接用实例链接访问
    )