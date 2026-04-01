import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os

# ---- CONFIG ----
st.title("📄 Conversational RAG Chatbot")
api_key = st.text_input("Enter GROQ API Key", type="password")

# ---- LOAD FILES ----
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

# ---- SESSION STORE ----
if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# ---- MAIN LOGIC ----
if api_key and uploaded_files:

    # 1. Load PDFs
    docs = []
    for file in uploaded_files:
        with open("temp.pdf", "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader("temp.pdf")
        docs.extend(loader.load())

    # 2. Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    # 3. Embeddings + Vector DB
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # 4. LLM
    llm = ChatGroq(model="openai/gpt-oss-120b", groq_api_key=api_key)

    # 5. History-aware retriever
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the question based on chat history."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    history_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    # 6. Answer chain
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer only from provided context. If unknown, say 'I don’t know'.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_retriever, qa_chain)

    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # ---- CHAT UI ----
    session_id = st.text_input("Session ID", value="default")

    user_input = st.chat_input("Ask something about your PDF...")

    if user_input:
        response = conversational_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )

        st.chat_message("user").write(user_input)
        st.chat_message("assistant").write(response["answer"])

else:
    st.info("Upload PDFs and enter API key to start.")