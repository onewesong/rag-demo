import streamlit as st
import openai
import dotenv
import os
import chromadb
from markitdown import MarkItDown
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib
import uuid

dotenv.load_dotenv()
md = MarkItDown()

chroma_client = chromadb.Client()

# 初始化或获取 collection
st.session_state.collection = chroma_client.get_or_create_collection(
    name="document_chunks",
    metadata={"description": "文档切片集合"}
)

if 'handled_files' not in st.session_state:
    st.session_state.handled_files = []

# 页面标题
st.header("💬 RAG Demo", divider="rainbow")
st.caption("🚀 基于 Streamlit、Chroma 和大模型 API 的智能问答系统")

@st.cache_data
def get_models():
    """获取可用的模型列表"""
    models = []
    try:
        for model in llm_client.models.list().data:
            if 'rerank' in model.id.lower() or 'bge' in model.id.lower():
                continue
            models.append(model.id)
    except Exception as e:
        st.error(f"获取模型列表失败: {e}")
        return []
    return models

def process_and_store_document(file):
    """处理并存储文档"""
    try:
        # 将文档转换为 Markdown
        content = md.convert(file).text_content
        
        # 切分文档
        chunks = text_splitter.split_text(content)
        
        # 生成文档ID（基于文件名的哈希）
        file_hash = hashlib.md5(file.name.encode()).hexdigest()
        
        # 存储每个切片
        chunk_ids = []
        chunk_docs = []
        chunk_metas = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{file_hash}_{i}"
            chunk_ids.append(chunk_id)
            chunk_docs.append(chunk)
            chunk_metas.append({
                "source": file.name,
                "chunk_index": i,
                "total_chunks": len(chunks)
            })
        
        # 批量添加到 Chroma
        st.session_state.collection.add(
            documents=chunk_docs,
            metadatas=chunk_metas,
            ids=chunk_ids,
        )
        
        return True, len(chunks), content
    except Exception as e:
        return False, 0, str(e)

@st.dialog("📚 数据库文档切片", width="large")
def show_chunks_dialog():
    """显示数据库中的所有文档切片"""
    try:
        collection_count = st.session_state.collection.count()
        
        if collection_count == 0:
            st.info("数据库中暂无文档切片")
            return
        
        # 获取所有文档切片
        all_data = st.session_state.collection.get()
        
        st.write(f"共有 **{collection_count}** 个文档切片")
        
        # 按文档分组显示
        docs_by_source = {}
        for i, metadata in enumerate(all_data['metadatas']):
            source = metadata['source']
            if source not in docs_by_source:
                docs_by_source[source] = []
            docs_by_source[source].append({
                'id': all_data['ids'][i],
                'document': all_data['documents'][i],
                'metadata': metadata
            })
        
        # 显示每个文档的切片
        for source, chunks in docs_by_source.items():
            with st.expander(f"📄 {source} ({len(chunks)} 个切片)", expanded=True):
                sorted_chunks = sorted(chunks, key=lambda x: x['metadata']['chunk_index'])
                
                # 将切片分成两列显示
                for i in range(0, len(sorted_chunks), 2):
                    col1, col2 = st.columns(2)
                    
                    # 左列
                    with col1:
                        chunk = sorted_chunks[i]
                        st.markdown(f"**切片 {chunk['metadata']['chunk_index'] + 1}/{chunk['metadata']['total_chunks']}**")
                        st.text_area(
                            f"ID: {chunk['id']}",
                            chunk['document'],
                            height=200,
                            key=chunk['id'],
                            disabled=True
                        )
                    
                    # 右列（如果还有切片）
                    with col2:
                        if i + 1 < len(sorted_chunks):
                            chunk = sorted_chunks[i + 1]
                            st.markdown(f"**切片 {chunk['metadata']['chunk_index'] + 1}/{chunk['metadata']['total_chunks']}**")
                            st.text_area(
                                f"ID: {chunk['id']}",
                                chunk['document'],
                                height=200,
                                key=chunk['id'],
                                disabled=True
                            )
                
                if len(sorted_chunks) > 0:
                    st.divider()
    
    except Exception as e:
        st.error(f"获取文档切片失败: {e}")

# 侧边栏配置
with st.sidebar:
    st.subheader("⚙️ 系统配置")
    
    # API 配置
    base_url = st.text_input("OpenAI Base URL", value=os.getenv("OPENAI_BASE_URL", ""))
    api_key = st.text_input(
        "OpenAI API Key", 
        value=os.getenv("OPENAI_API_KEY", ""), 
        type="password", 
        help="获取 API Key: [白山大模型](https://ai.baishan.com/auth/login?referralCode=ttXv0P1zRH)"
    )
    
    if not api_key:
        st.error("⚠️ 请输入 OpenAI API Key")
        st.stop()
    
    llm_client = openai.OpenAI(base_url=base_url, api_key=api_key)
    
    # 模型选择
    model = st.selectbox("选择模型", get_models())
    
    # 检索参数配置
    st.divider()
    st.subheader("🔍 检索参数")
    n_results = st.slider("检索文档数量", min_value=1, max_value=10, value=3)
    
    # 文档切片参数配置
    st.divider()
    st.subheader("✂️ 文档切片参数")
    chunk_size = st.slider(
        "切片大小", 
        min_value=100, 
        max_value=2000, 
        value=1000, 
        step=100,
        help="每个切片的字符数"
    )
    chunk_overlap = st.slider(
        "切片重叠", 
        min_value=0, 
        max_value=500, 
        value=200, 
        step=50,
        help="切片之间的重叠字符数"
    )
    
    # 分隔符配置
    with st.expander("🔧 高级设置 - 分隔符配置"):
        separators_input = st.text_area(
            "分隔符列表（每行一个）",
            value="\\n\\n\n\\n\n。\n！\n？\n.\n!\n?\n \n",
            height=150,
            help="文本切片时使用的分隔符，按优先级从高到低排列。支持转义字符，如 \\n 表示换行"
        )
    
    separators = []
    for line in separators_input.strip().split('\n'):
        if line:
            # 处理转义字符
            sep = line.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
            separators.append(sep)
    
    # 如果没有分隔符，使用默认值
    if not separators:
        separators = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]

    # 初始化文本切分器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=separators,
    )

    # 文档上传
    st.divider()
    st.subheader("📁 文档上传")
    uploaded_files = st.file_uploader(
        "上传文档", 
        type=["pdf", "docx", "txt", "md", "pptx", "xlsx"],
        accept_multiple_files=True,
        help="支持 PDF、Word、文本、Markdown、PPT、Excel 等格式"
    )
    if uploaded_files:
        for file in uploaded_files:
            if file.name in st.session_state.handled_files:
                continue
            st.session_state.handled_files.append(file.name)
            with st.spinner(f"正在处理 {file.name}..."):
                success, chunks_count, result = process_and_store_document(file)
                
                if success:
                    st.success(f"✅ {file.name} 已处理（共 {chunks_count} 个切片）")
                    with st.expander(f"📄 查看 {file.name} 内容"):
                        st.text(result[:1000] + "..." if len(result) > 1000 else result)
                else:
                    st.error(f"❌ {file.name} 处理失败: {result}")
    
    # 数据库统计
    st.divider()
    st.subheader("📊 数据库统计")
    try:
        collection_count = st.session_state.collection.count()
        st.metric("文档切片数", collection_count)
    except:
        st.metric("文档切片数", "N/A")
    
    # 查看文档切片按钮
    if st.button("👀 查看文档切片"):
        show_chunks_dialog()
    
    # 清空数据库
    if st.button("🗑️ 清空数据库"):
        chroma_client.delete_collection("document_chunks")
        st.session_state.handled_files = []
        st.toast("数据库已清空")
        st.rerun()
    # 聊天处理
    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
    # 调试选项
    st.divider()
    if st.toggle("🐛 显示调试信息"):
        st.write("**聊天历史：**", st.session_state.get('messages', []))
        st.write("**切片配置：**")
        st.json({
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "separators_count": len(separators),
            "separators": [repr(s) for s in separators]
        })

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示聊天历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        # 如果有检索结果，显示来源
        if "sources" in message:
            with st.expander("📚 参考来源"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**来源 {i}:** {source['metadata']['source']} (切片 {source['metadata']['chunk_index']+1}/{source['metadata']['total_chunks']})")
                    st.text(source['document'][:200] + "..." if len(source['document']) > 200 else source['document'])

# 聊天输入
if prompt := st.chat_input("请输入您的问题..."):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # 检索相关文档
    with st.spinner("🔍 正在检索相关文档..."):
        results = st.session_state.collection.query(
            query_texts=[prompt],
            n_results=n_results,
        )
        
        # 提取检索到的文档
        retrieved_docs = []
        if results['documents'] and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                retrieved_docs.append({
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
        
        # 构建上下文
        context = "\n\n".join([doc['document'] for doc in retrieved_docs])
        
        # 构建提示词
        system_prompt = f"""你是一个智能助手，请基于以下检索到的文档内容回答用户的问题。

检索到的相关文档：
<context>
{context}
</context>

请根据上述文档内容回答问题。如果文档中没有相关信息，请如实告知用户。"""

    # 调用 LLM 生成答案（流式输出）
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    # 显示答案（流式）
    with st.chat_message("assistant"):
        # 创建一个占位符用于流式输出
        message_placeholder = st.empty()
        full_response = ""
        
        # 调用 LLM（流式）
        stream = llm_client.chat.completions.create(
            model=model, 
            messages=messages,
            temperature=0.7,
            stream=True,  # 启用流式输出
        )
        
        # 逐步接收并显示响应
        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
        
        # 显示完整响应（移除光标）
        message_placeholder.markdown(full_response)
        
        # 显示参考来源
        if retrieved_docs:
            for i, doc in enumerate(retrieved_docs, 1):
                with st.expander(f"**来源 {i}:** {doc['metadata']['source']} (切片 {doc['metadata']['chunk_index']+1}/{doc['metadata']['total_chunks']})"):
                    if doc['distance'] is not None:
                        st.caption(f"相似度距离: {doc['distance']:.4f}")
                    st.text(doc['document'])
    
    # 添加助手消息到历史记录
    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response,
        "sources": retrieved_docs
    })
        
        
