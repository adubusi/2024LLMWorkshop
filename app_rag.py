import streamlit as st
import pypdf
import ollama
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter

# 配置Streamlit页面
st.title("PDF RAG问答系统 - Ollama mistral模型")

# 上传PDF文件
uploaded_file = st.file_uploader("请上传一个PDF文件", type="pdf")

if uploaded_file is not None:
    # 从PDF中提取文本
    reader = pypdf.PdfReader(uploaded_file)
    pdf_text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        pdf_text += page.extract_text()

    # 将PDF文本进行分段，以便创建文档列表
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(pdf_text)

    # 将文本片段转换为Document对象
    documents = [Document(page_content=chunk) for chunk in chunks]

    # 使用HuggingFace的嵌入模型
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    # 构建FAISS向量数据库
    vector_store = FAISS.from_documents(documents, embeddings)

    st.write("PDF文本已成功索引。")

    # 输入问题
    question = st.text_input("请输入你的问题：")

    if question:
        # 从向量数据库中检索与问题最相关的段落
        docs = vector_store.similarity_search(question, k=2)
        retrieved_text = " ".join([doc.page_content for doc in docs])

        st.write("检索到的相关内容：")
        st.write(retrieved_text)

        # 使用Ollama Phi3模型回答问题
        ollama_response = ollama.chat(model="mistral", messages=[
            {"role": "system", "content": "你是一个帮助用户从文档中回答问题的助手。"},
            {"role": "user", "csontent": f"基于以下内容回答问题：{retrieved_text} 问题是：{question}"}
        ])

        # 打印完整的响应
        st.write("完整响应：")
        st.write(ollama_response)

        # 尝试提取答案
        try:
            answer = ollama_response['choices'][0]['message']['content']
            st.write("模型回答：")
            st.write(answer)
        except KeyError as e:
            st.write(f"无法生成答案，可能是响应格式有问题: {e}")