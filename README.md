# 📄 Conversational RAG Chatbot (Multi-PDF + Chat History)

An AI-powered conversational chatbot that allows users to upload multiple PDFs and ask questions about their content using Retrieval-Augmented Generation (RAG).

---

## 🚀 Features

* 📂 Upload multiple PDF documents
* 💬 Ask questions in a conversational manner
* 🧠 Maintains chat history for better follow-up answers
* 🔍 Retrieves relevant information using vector search (ChromaDB)
* ⚡ Fast inference using Groq LLM
* 🆓 Free embeddings using HuggingFace

---

## 🧠 How It Works

1. Upload PDF documents
2. Text is extracted and split into smaller chunks
3. Chunks are converted into embeddings (vector representation)
4. Stored in ChromaDB (vector database)
5. User asks a question
6. Relevant chunks are retrieved
7. LLM generates answer based on context
8. Chat history improves follow-up responses

---

## 🛠️ Tech Stack

* Streamlit (UI)
* LangChain (RAG pipeline)
* ChromaDB (Vector Database)
* HuggingFace Embeddings
* Groq LLM (openai/gpt-oss-120b)

---

## 📂 Project Structure

```
├── main1.py
├── requirements.txt
├── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add API Key

Create a `.env` file:

```
GROQ_API_KEY=your_api_key_here
```

---

### 4. Run the app

```bash
streamlit run app.py
```

---

## 💡 Example Use Cases

* 📚 Study assistant for PDFs
* 📄 Resume or document analysis
* 📊 Research paper Q&A
* 🏢 Business document insights

---

## 🎯 Key Highlights

* Implemented **history-aware retriever** for contextual understanding
* Built **conversational memory system** for multi-turn interactions
* Designed efficient **RAG pipeline** using LangChain
* Optimized for performance using **Groq LLM + HuggingFace embeddings**

---

## 📌 Future Improvements

* Add source citations for answers
* Improve UI with chat interface enhancements
* Deploy on Streamlit Cloud

---

## 🙌 Author

**Girish Potlada Yellapu**
Master’s in Business Analytics & AI | Aspiring AI Engineer

---

## ⭐ If you like this project, give it a star!
