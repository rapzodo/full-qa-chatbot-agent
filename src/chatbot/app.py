import os

import streamlit as st
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

GROQ_MODELS = ["gemma2-9b-it", "llama-3.1-8b-instant", "llama-3.3-70b-versatile",
               "meta-llama/llama-guard-4-12b", "deepseek-r1-distill-llama-70b"]

load_dotenv()


def create_prompt():
    return ChatPromptTemplate(
        [
            ("system", "You are a helpful assistant, respond to the user questions"),
            ("human", "Question:{question}")
        ]
    )


def create_prompt_for_rag():
    return ChatPromptTemplate.from_template(
        """
        Answer the questions based on provided context only.
        Context: {context}
        
        Question: {input}
        """
    )


def get_groc_model(max_tokens, model_id, temperature):
    return ChatGroq(model=model_id, api_key=os.getenv("GROQ_API_KEY"), temperature=temperature, max_tokens=max_tokens)


def generate_response(user_question, model_id, temperature, max_tokens):
    llm = get_groc_model(max_tokens, model_id, temperature)
    chain = create_prompt() | llm | StrOutputParser()
    return chain.invoke({"question": user_question})


def create_vector_embeddings():
    if "vector_store" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("files")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.chunks = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vector_store = Chroma.from_documents(st.session_state.docs, st.session_state.embeddings)


def rag_search_ui(model, temp, max_tokens):
    uploader = st.file_uploader("Upload files", ["pdf", "word", "csv"], accept_multiple_files=True)
    if uploader:
        for uploaded_file in uploader:
            save_path = os.path.join("files", uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success("Files uploaded and saved")
        create_vector_embeddings()
        st.write("Docs loaded!")

    # Use a form to control when the query is submitted
    with st.form(key="rag_form", clear_on_submit=True):
        user_prompt = st.text_input("Enter your question")
        submit_button = st.form_submit_button("Search")

    if submit_button and user_prompt:
        doc_chain = create_stuff_documents_chain(
            get_groc_model(model_id=model, temperature=temp, max_tokens=max_tokens), prompt=create_prompt_for_rag())
        retriever = st.session_state.vector_store.as_retriever()
        response = create_retrieval_chain(retriever, doc_chain).invoke({"input": user_prompt})

        st.session_state.last_response = response
        st.write(response["answer"])
        with st.expander("Sources"):
            for index, doc in enumerate(st.session_state.last_response["context"]):
                st.write(doc)
                st.write('------------')


def chat_ui():
    model, selected_max_tokens, selected_temperature = set_base_ui()
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""


    with st.form(key="rag_form", clear_on_submit=True):
        user_prompt = st.text_input("Enter your question")
        submit_button = st.form_submit_button("Send")

    if submit_button or user_prompt:
        if user_prompt:
            response = generate_response(user_prompt, model_id=model, temperature=selected_temperature,
                                         max_tokens=selected_max_tokens)
            st.write(response)
            st.session_state.user_input = ""
        else:
            st.write("I did not get you question, did you ask any ?")


def set_base_ui():
    st.title("Q&A Chatbot")
    st.sidebar.title("settings")
    model = st.sidebar.selectbox("Select a model", GROQ_MODELS)
    selected_temperature = st.sidebar.slider("temperature", min_value=0.0, max_value=1.0, value=0.7)
    selected_max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)
    st.write("I'm Danilo's Chatbot. How can I help you")
    return model, selected_max_tokens, selected_temperature


def set_rag_ui():
    model, selected_max_tokens, selected_temperature = set_base_ui()
    rag_search_ui(model, selected_temperature, selected_max_tokens)


mode = st.sidebar.radio("Mode",["Chat", "RAG"])
if mode == "RAG":
    set_rag_ui()
else:
    chat_ui()