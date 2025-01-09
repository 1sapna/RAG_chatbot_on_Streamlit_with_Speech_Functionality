#=================
# Import Libraries
#=================
import streamlit as st
import pandas as pd
import os
from langchain.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import HuggingFaceChat
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from io import BytesIO
from uuid import uuid4

#=================
# Background Image, Chatbot Title, and Logo
#=================
page_bg_img = '''
<style>
.stApp {
background-image: linear-gradient(rgba(255, 255, 255, 0.5), rgba(255, 255, 255, 0.5)), url("https://imageio.forbes.com/specials-images/imageserve/6271151bcd7b0b7ffd1fa4e2/Artificial-intelligence-robot/960x0.jpg");
background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("Gen AI RAG Chatbot")
try:
    image_url = "logo-new.png"
    st.sidebar.image(image_url, caption="", use_column_width=True)
except:   
    image_url = "https://static.vecteezy.com/system/resources/previews/010/794/341/non_2x/purple-artificial-intelligence-technology-circuit-file-free-png.png"
    st.sidebar.image(image_url, caption="", use_column_width=True)

#=================
# File Upload and Format Selection
#=================
file_format = st.sidebar.selectbox("Select File Format", ["CSV", "PDF", "TXT"])
if file_format == "TXT":
    file_format = "plain"

uploaded_files = st.sidebar.file_uploader("Upload a file", type=["csv", "txt", "pdf"], accept_multiple_files=True)

def validateFormat(file_format, uploaded_files):
    for file in uploaded_files:
        if str(file_format).lower() not in str(file.type).lower():
            return False
    return True

def save_uploadedfile(uploadedfile):
    with open(os.path.join(uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved File")

#=================
# Helper Functions
#=================
def history_func(answer, q):
    if 'history' not in st.session_state:
        st.session_state.history = ''
    value = f'Q: {q} \nA: {answer}'
    st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
    st.text_area(label='Chat History', value=st.session_state.history, key='history', height=400)

def CSVAnalysis(uploaded_file):
    df = pd.read_csv(uploaded_file)
    st.header("Dataframe Preview")
    st.write(df)
    save_uploadedfile(uploaded_file)
    user_query = st.text_input('Enter your query')
    if st.button("Answer My Question"):
        st.write("Processing query...")
        # For demo purposes, returning a mock response
        st.text_area('LLM Answer: ', value="Mock answer for query: " + user_query, height=400)
        history_func("Mock answer", user_query)

def MergePDFAnalysis(uploaded_files):
    raw_text = ''
    for file in uploaded_files:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    texts = text_splitter.split_text(raw_text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    docsearch = FAISS.from_texts(texts, embeddings)
    question = st.text_input("Enter your question")
    if st.button("Answer My Question"):
        docs = docsearch.similarity_search(question)
        answer = "Mock answer from retrieved documents."
        st.text_area('LLM Answer: ', value=answer, height=400)
        history_func(answer, question)

#=================
# Main Logic
#=================
if uploaded_files:
    if validateFormat(file_format, uploaded_files):
        if file_format == "CSV":
            for file in uploaded_files:
                CSVAnalysis(file)
        elif file_format == "PDF":
            MergePDFAnalysis(uploaded_files)
        else:
            st.write("File format not supported.")
