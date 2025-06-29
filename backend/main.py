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
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.agents import Tool, create_tool_calling_agent, AgentExecutor

import os

# ========== Load Environment Variables ==========
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY") or "your_fallback_api_key_here"

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

# ========== MBA Application Tool ==========
def collect_mba_application(data: str) -> str:
    with open("mba_applications.txt", "a", encoding="utf-8") as f:
        f.write(data + "\n\n")
    return "✅ Your MBA application has been received. We'll contact you soon."

mba_application_tool = Tool(
    name="ApplyMBA",
    func=collect_mba_application,
    description=(
        "Use this tool when the user says they want to apply for MBA. "
        "Ask their name, email, phone number, address, qualification, and age. "
        "Once collected, pass all details as a single string to this tool."
    )
)

# ========== LLM + Prompt ==========
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=API_KEY)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant for college queries. Use tools when needed."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# ========== Agent Setup ==========
agent = create_tool_calling_agent(
    llm=llm,
    tools=[mba_application_tool],
    prompt=prompt,
)

agent_executor = AgentExecutor(agent=agent, tools=[mba_application_tool], verbose=True)

# ========== Routes ==========
@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat")
async def chat_with_bot(query: str = Form(...)):
    try:
        context_docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in context_docs])

        full_query = f"{query}\n\nRelevant college context:\n{context}"
        response = agent_executor.invoke({"input": full_query, "chat_history": []})
        return JSONResponse(content={"response": response["output"]})
    except Exception as e:
        return JSONResponse(content={"response": "Sorry, something went wrong."})

# ========== Run ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
