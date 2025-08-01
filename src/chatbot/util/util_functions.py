import os

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings


def document_upload():
    uploader = st.file_uploader("Upload files", ["pdf", "word", "csv"], accept_multiple_files=True)
    if uploader:
        for uploaded_file in uploader:
            save_path = os.path.join("files", uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success("Files uploaded and saved")
        create_vector_embeddings()
        st.write("Docs loaded!")


def create_vector_embeddings():
    if "vector_store" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("files")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.chunks = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vector_store = Chroma.from_documents(st.session_state.docs, st.session_state.embeddings)



def get_groc_model(max_tokens, model_id, temperature):
    return ChatGroq(model=model_id, api_key=os.getenv("GROQ_API_KEY"), temperature=temperature, max_tokens=max_tokens)