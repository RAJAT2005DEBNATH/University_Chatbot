import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

# Modern LangChain & Gemini Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. LOAD ENVIRONMENT
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("API Key not found! Please ensure 'GOOGLE_API_KEY' is set in your .env file.")
    st.stop()

# 2. INITIALIZE GEMINI 2.5 FLASH
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.3)

# 3. DATA PREPROCESSING (RAG Setup)
@st.cache_resource
def initialize_rag():
    file_path = "university_docs.txt"
    
    if not os.path.exists(file_path):
        st.error(f"Critical Error: '{file_path}' not found.")
        st.stop()

    try:
        # Load raw university text data
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        
        # Chunking (~500 tokens as per your assignment)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        
        # LATEST 2026 EMBEDDING MODEL
        # Gemini Embedding 2 supports text, PDF, and multimodal inputs
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2-preview")
        
        # Store in Vector DB (FAISS)
        vector_db = FAISS.from_documents(texts, embeddings)
        return vector_db
    except Exception as e:
        st.error(f"Failed to process {file_path}: {e}")
        st.stop()

vector_db = initialize_rag()
retriever = vector_db.as_retriever()

# RAG Chain Setup
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 4. STREAMLIT UI
st.title("🏛️ University Student Support Agent")

with st.sidebar:
    st.header("Raw Feedback Data")
    if os.path.exists("student_feedback.csv"):
        df = pd.read_csv("student_feedback.csv")
        st.dataframe(df)
    else:
        st.error("student_feedback.csv missing!")

tab1, tab2, tab3 = st.tabs(["Ask Queries", "Summarize Feedback", "Generate Report"])

# TOOL 1: RAG Tool
with tab1:
    user_query = st.text_input("Ask about rules, courses, or events:")
    if user_query:
        with st.spinner("Searching documents..."):
            response = rag_chain.invoke(user_query)
            st.write("**Agent Response:**", response)

# TOOL 2: Summarization Tool
with tab2:
    if st.button("Summarize Top 3 Complaints"):
        feedback_raw = "\n".join(df['Feedback'].astype(str).tolist())
        sum_prompt = f"Summarize the top 3 student complaints from this feedback: {feedback_raw}"
        summary = llm.invoke(sum_prompt)
        st.info(summary.content)

# TOOL 3: Report Generator Tool
with tab3:
    if st.button("Generate Academic Feedback Report"):
        feedback_raw = "\n".join(df['Feedback'].astype(str).tolist())
        rep_prompt = f"Generate a structured university report with 'Positives' and 'Negatives' based on: {feedback_raw}"
        report = llm.invoke(rep_prompt)
        st.success(report.content)