import streamlit as st
import openai
import dotenv
import os

dotenv.load_dotenv()

if 'handled_files' not in st.session_state:
    st.session_state.handled_files = []

# 页面标题
st.header("💬 RAG Demo", divider="rainbow")
st.caption("🚀 基于 Streamlit、OpenAI 向量库 和大模型 API 的智能问答系统")

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
    """上传文件到 OpenAI 向量库并等待索引完成。"""
    try:
        # 创建或获取向量库
        if "vector_store_id" not in st.session_state or not st.session_state.vector_store_id:
            vs = llm_client.vector_stores.create(name="document_store")
            st.session_state.vector_store_id = vs.id

        # 上传并索引到向量库
        llm_client.vector_stores.files.upload_and_poll(
            vector_store_id=st.session_state.vector_store_id,
            file=file,
        )

        return True, None
    except Exception as e:
        return False, str(e)

@st.dialog("📚 向量库文件与内容", width="large")
def show_chunks_dialog():
    """显示向量库中的文件及解析内容（按文件展示）。"""
    try:
        if "vector_store_id" not in st.session_state or not st.session_state.vector_store_id:
            st.info("向量库为空，尚未上传任何文件。")
            return

        # 列出向量库文件
        files_page = llm_client.vector_stores.files.list(vector_store_id=st.session_state.vector_store_id, limit=100)
        files_list = getattr(files_page, "data", files_page)

        if not files_list:
            st.info("向量库中暂无文件")
            return

        st.write(f"共有 **{len(files_list)}** 个文件")

        for f in files_list:
            with st.expander(f"📄 {getattr(f, 'filename', getattr(f, 'id', 'file'))} - 状态: {getattr(f, 'status', 'unknown')}"):
                try:
                    contents_page = llm_client.vector_stores.files.content(
                        file_id=f.id,
                        vector_store_id=st.session_state.vector_store_id,
                    )
                    contents = getattr(contents_page, "data", contents_page)
                    # 每个内容项包含 text
                    for idx, item in enumerate(contents[:10]):  # 仅展示前 10 段，避免过长
                        st.text_area(
                            f"内容片段 {idx+1}",
                            getattr(item, "text", ""),
                            height=200,
                            key=f"{f.id}_content_{idx}",
                            disabled=True,
                        )
                except Exception as ex:
                    st.warning(f"读取文件内容失败: {ex}")

    except Exception as e:
        st.error(f"获取向量库文件失败: {e}")

# 侧边栏配置
with st.sidebar:
    st.subheader("⚙️ 系统配置")
    
    # API 配置
    base_url = st.text_input("OpenAI Base URL", value=os.getenv("OPENAI_BASE_URL", "https://api.edgefn.net/v1/"))
    api_key = st.text_input(
        "OpenAI API Key", 
        value=os.getenv("OPENAI_API_KEY", ""), 
        type="password", 
        help="可点击此链接获取 API Key: [白山大模型](https://ai.baishan.com/auth/login?referralCode=ttXv0P1zRH), 注册即送150元"
    )
    
    if not api_key:
        st.error("⚠️ 请输入 OpenAI API Key")
        st.info("可点击此链接获取 API Key: [白山大模型](https://ai.baishan.com/auth/login?referralCode=ttXv0P1zRH), 注册即送150元")
        st.stop()
    
    llm_client = openai.OpenAI(base_url=base_url, api_key=api_key)
    
    # 向量库选择/创建
    st.divider()
    st.subheader("🗄️ 向量库")
    try:
        vs_page = llm_client.vector_stores.list(limit=50)
        vs_list = getattr(vs_page, "data", vs_page)
        vs_options = [f"{(vs.name or vs.id)} ({vs.id})" for vs in vs_list]
        if vs_options:
            selected = st.selectbox(
                "选择向量库",
                options=list(range(len(vs_options))),
                format_func=lambda i: vs_options[i],
            )
            if selected is not None:
                st.session_state.vector_store_id = vs_list[selected].id
        new_vs_name = st.text_input("新向量库名称", value="")
        if st.button("➕ 创建向量库"):
            created_vs = llm_client.vector_stores.create(name=new_vs_name or "document_store")
            st.session_state.vector_store_id = created_vs.id
            st.toast("已创建向量库")
            st.rerun()
    except Exception as e:
        st.warning(f"加载向量库列表失败: {e}")
    
    # 模型选择
    model = st.selectbox("选择模型", get_models())
    
    # 检索参数配置
    st.divider()
    st.subheader("🔍 检索参数")
    n_results = st.slider("检索文档数量", min_value=1, max_value=20, value=5)
    
    #（已移除本地切片配置，改为服务端自动解析与切片）

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
                success, error = process_and_store_document(file)

                if success:
                    st.success(f"✅ {file.name} 已上传并索引至 OpenAI 向量库")
                else:
                    st.error(f"❌ {file.name} 上传/索引失败: {error}")
    
    # 向量库统计
    st.divider()
    st.subheader("📊 向量库统计")
    try:
        if "vector_store_id" in st.session_state and st.session_state.vector_store_id:
            files_page = llm_client.vector_stores.files.list(vector_store_id=st.session_state.vector_store_id, limit=100)
            files_list = getattr(files_page, "data", files_page)
            st.metric("向量库文件数", len(files_list))
        else:
            st.metric("向量库文件数", 0)
    except Exception:
        st.metric("向量库文件数", "N/A")
    
    # 查看文件与内容按钮
    if st.button("👀 查看向量库内容"):
        show_chunks_dialog()
    
    # 清空向量库
    if st.button("🗑️ 清空向量库"):
        try:
            if "vector_store_id" in st.session_state and st.session_state.vector_store_id:
                llm_client.vector_stores.delete(st.session_state.vector_store_id)
            st.session_state.vector_store_id = None
            st.session_state.handled_files = []
            st.toast("向量库已清空")
            st.rerun()
        except Exception as e:
            st.error(f"清空向量库失败: {e}")
    # 聊天处理
    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
    # 调试选项
    st.divider()
    if st.toggle("🐛 显示调试信息"):
        st.write("**聊天历史：**", st.session_state.get('messages', []))

# 显示参考来源
def display_retrieved_docs(retrieved_docs):
    if not retrieved_docs:
        return
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.get('metadata', {}).get('source') or doc.get('metadata', {}).get('filename') or "未知来源"
        header = f"**来源 {i}:** {source}"
        with st.expander(header):
            score = doc.get('score')
            distance = doc.get('distance')
            if score is not None:
                st.caption(f"相似度得分: {score:.4f}")
            elif distance is not None:
                st.caption(f"相似度距离: {distance:.4f}")
            st.text(doc['document'])

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示聊天历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "sources" in message:
            display_retrieved_docs(message["sources"])


# 聊天输入
if prompt := st.chat_input("请输入您的问题..."):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # 检索相关文档（使用 OpenAI 向量库搜索）
    with st.spinner("🔍 正在检索相关文档..."):
        retrieved_docs = []
        if "vector_store_id" in st.session_state and st.session_state.vector_store_id:
            try:
                search_page = llm_client.vector_stores.search(
                    st.session_state.vector_store_id,
                    query=prompt,
                    max_num_results=n_results,
                )
                search_results = getattr(search_page, "data", search_page)
                for item in search_results:
                    # item.content 是列表，拼接为文本
                    texts = []
                    try:
                        for c in getattr(item, "content", [])[:3]:
                            t = getattr(c, "text", None)
                            if t:
                                texts.append(t)
                    except Exception:
                        pass
                    doc_text = "\n\n".join(texts) if texts else ""
                    if not doc_text:
                        # 兜底：若无 content，则跳过
                        continue
                    retrieved_docs.append({
                        'document': doc_text,
                        'metadata': {
                            'source': getattr(item, 'filename', 'unknown'),
                        },
                        'score': getattr(item, 'score', None),
                    })
            except Exception as e:
                st.warning(f"向量库搜索失败，改为无检索回答：{e}")
        
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
        
        display_retrieved_docs(retrieved_docs)
    
    # 添加助手消息到历史记录
    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response,
        "sources": retrieved_docs
    })
        
        
