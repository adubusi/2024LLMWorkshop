import streamlit as st
import pypdf
import ollama  # 使用ollama库直接调用模型

# 配置Streamlit页面
st.title("PDF问答系统 - Ollama Phi3模型")

# 上传PDF文件
uploaded_file = st.file_uploader("请上传一个PDF文件", type="pdf")

if uploaded_file is not None:
    # 从PDF中提取文本
    reader = pypdf.PdfReader(uploaded_file)
    pdf_text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        pdf_text += page.extract_text()

    # 显示PDF中的文本
    st.write("PDF内容：")
    st.text_area("PDF内容", value=pdf_text, height=200)

    # 输入问题
    question = st.text_input("请输入你的问题：")

    if question:
        # 使用Ollama Phi3模型回答问题
        # 调用模型
        response = ollama.chat(model="phi3", messages=[{"role": "system", "content": pdf_text},
                                                       {"role": "user", "content": question}])

        # 打印完整的响应
        st.write("完整响应内容：")
        st.write(response)  # 打印响应，查看其中的结构
