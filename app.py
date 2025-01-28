import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS  # Changed to FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import os

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

pdfs_directory = 'My_first_proj/pdfs/'

# Create directory if it doesn't exist
os.makedirs(pdfs_directory, exist_ok=True)

# Initialize embeddings
embeddings = OllamaEmbeddings(model="deepseek-r1:14b")

# Initialize FAISS vector store
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = FAISS.from_texts([""], embeddings)

model = OllamaLLM(model="deepseek-r1:14b")

def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

@st.cache_data  # Add caching for better performance
def process_documents(documents):
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    return FAISS.from_texts(texts, embeddings, metadatas=metadatas)

def retrieve_docs(query, vector_store):
    return vector_store.similarity_search(query, k=4)  # Limit to top 4 results for speed

def answer_question(question, documents):
    # Only use the most relevant parts of the context
    context = "\n\n".join([doc.page_content for doc in documents[:2]])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})

# UI
uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=False
)

if uploaded_file:
    with st.spinner('Processing PDF...'):
        upload_pdf(uploaded_file)
        documents = load_pdf(pdfs_directory + uploaded_file.name)
        chunked_documents = split_text(documents)
        # Update vector store
        st.session_state.vector_store = process_documents(chunked_documents)
        st.success('PDF processed successfully!')

question = st.chat_input()
if question:
    st.chat_message("user").write(question)
    with st.spinner('Searching...'):
        related_documents = retrieve_docs(question, st.session_state.vector_store)
        answer = answer_question(question, related_documents)
        st.chat_message("assistant").write(answer)