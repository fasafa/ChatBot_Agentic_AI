import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader





def create_faiss_vector_store(filepath: str, api_key: str) -> FAISS:
    os.environ["GOOGLE_API_KEY"] = api_key
    loader = TextLoader(filepath, encoding="utf-8")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    db = FAISS.from_documents(texts, embeddings)
    db.save_local("faiss_index")
    return db

def load_faiss_vector_store(api_key: str) -> FAISS:
    os.environ["GOOGLE_API_KEY"] = api_key
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)


