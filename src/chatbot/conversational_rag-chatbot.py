import random

import streamlit as st
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnableConfig

from src.chatbot.app import GROQ_MODELS
from src.chatbot.util.util_functions import get_groc_model, create_vector_embeddings, document_upload

st.title("Conversational RAG with PDF and chat history")
st.write("Upload PDF and ask questions about the files")

llm = get_groc_model(max_tokens=1000, model_id=GROQ_MODELS[0], temperature=0.6)

if "session_id" not in st.session_state:
    st.session_state.session_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))

if "store" not in st.session_state:
    st.session_state.store = {}

document_upload()

preamble = """
    Given the chat history and the previous context, write the responses in the form of a question
"""
contextualized_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", preamble),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

create_vector_embeddings()

history_aware_retriever = create_history_aware_retriever(llm, st.session_state.vector_store.as_retriever(),
                                                         contextualized_prompt)

system_prompt = """
    You are a helpful assistant,
    answer the questions based on the context.
    If you dont know the answer say: I don't know. I can't know everything.
    
    Context: {context}
"""

initial_question = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, initial_question)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = InMemoryChatMessageHistory()
    return st.session_state.store[session]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain, get_session_history, input_messages_key="input", history_messages_key="chat_history",
    output_messages_key="answer"
)

user_input = st.text_input("Question:")
if user_input:
    session_history = get_session_history(st.session_state.session_id)
    response = conversational_rag_chain.invoke(
        {"input": user_input},
        config=RunnableConfig(configurable={"session_id": st.session_state.session_id})
    )
    st.write(st.session_state.store)
    st.write("Assistant:", response["answer"])
    st.write("Chat History", session_history.messages)
