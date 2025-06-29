# 🎓 Agentic Chatbot for Educational Institution

This project is an intelligent **chatbot assistant** built for an educational institution. It helps students and parents with queries about the college and also collects application details for the MBA program using a LangChain **agent + tool** architecture.

---

## ✨ Features

- Answers college-related questions using your custom data
- Agent powered by **LangChain + Gemini LLM**
- Stores and retrieves documents with **FAISS vector store**
- Handles tool-based workflows (e.g., MBA application form)
- Clean **FastAPI backend** + minimal HTML frontend
- Stores MBA application details to a text file

---

## ⚙️ Tech Stack

- **FastAPI** – Web API framework  
- **LangChain** – Agent framework for LLM integration  
- **Gemini (via Google Generative AI)** – LLM provider  
- **FAISS** – Vector store for similarity search  
- **Jinja2** – HTML rendering  
- **HTML/CSS/JS** – Frontend (chat interface)

---

## 🗂 Folder Structure

ChatBot_Agentic_AI/
│
├── backend/
│ ├── main.py # Main FastAPI server with agent + tool
│ ├── utils.py # (optional) Helper functions
│ ├── mba_applications.txt # File to store MBA form data
│ ├── faiss_index/ # Stored FAISS vector store
│
├── frontend/
│ ├── index.html # Frontend chatbot UI
│ └── style.css # Optional CSS styling
│
├── data.txt # Text document with college info
├── requirements.txt # Python package dependencies
└── README.md # You're reading this!



---

##  How It Works

1. **Startup**:
   - `data.txt` is split into chunks and embedded into FAISS.
   - A retriever fetches relevant chunks based on query similarity.

2. **Agent**:
   - User query is routed through an **agent**.
   - The agent uses a Gemini-powered LLM.
   - If it detects MBA interest, it calls the `ApplyMBA` tool to collect details.

3. **MBA Application Tool**:
   - Stores name, email, phone, address, qualification, and age to a file.

---

## 🚀 Getting Started

### ✅ Prerequisites

- Python 3.9+
- Google Gemini API key
- `data.txt` file with your institution's info

---

### 🛠️ Installation

1. **Clone the repo**

```bash
git clone https://github.com/fasafa/ChatBot_Agentic_AI.git
cd ChatBot_Agentic_AI

2. **Create virtual environment**

   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate







