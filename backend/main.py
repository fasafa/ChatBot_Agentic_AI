from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

import os

# ========== Load Environment Variables ==========
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY") or "AIzaSyDRYFJnN0kIEkkJHGdk-3trviIG5_WWR78"  # fallback if .env not set

# ========== FastAPI Setup ==========
app = FastAPI()
app.mount("/static", StaticFiles(directory="../frontend"), name="static")
templates = Jinja2Templates(directory="../frontend")

# ========== Load & Index Text Data ==========
if not os.path.exists("faiss_index"):
    loader = TextLoader("data.txt", encoding="ISO-8859-1")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")
else:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ========== Prompt Template ==========
prompt_template = ChatPromptTemplate.from_template("""
You are a helpful assistant for college-related queries. Use the following context to answer:

Context:
{context}

Question:
{question} #system

Please respond in no more than 1 line. Be clear and direct.
""")
output_parser = StrOutputParser()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=API_KEY)
chain = prompt_template | llm | output_parser 

# ========== Endpoints ==========
@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat")
async def chat_with_bot(query: str = Form(...)):
    try:
        context_docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in context_docs])

        response = chain.invoke({"context": context, "question": query})
        return JSONResponse(content={"response": response})
    except Exception as e:
        return JSONResponse(content={"response": "Sorry, I don't have enough information to answer that."})

# ========== Run App ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
