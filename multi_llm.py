import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
from apple.search import RealAppleWebScraper

# Load environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0, api_key=GROQ_API_KEY)

# Load vector DB (assuming built already)
vectordb = Chroma(persist_directory="./chroma_db", embedding_function=None)  # set embedding fn
retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3})

# Prompt
prompt = ChatPromptTemplate.from_template("""
You are an expert Apple Watch assistant. 
Answer user questions based on the given context. If not enough info, just say "NO_CONTEXT".
Context: {context}
Question: {question}
Answer:
""")

# QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

# Scraper Fallback
scraper = RealAppleWebScraper()

# Streamlit UI
st.title("⌚ Apple Watch Assistant")
query = st.text_input("Ask about Apple Watch:")

if query:
    # Step 1: Try RAG
    response = qa_chain.run(query)

    # Step 2: If local context missing, fallback to scraper
    if "NO_CONTEXT" in response:
        response = scraper.fallback_query(query)

    st.write(response)
