from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# Load and split the PDF into chunks
loader = PyPDFLoader("4.pdf")  
raw_docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = splitter.split_documents(raw_docs)

# Create embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")


# Setup Chroma DB
db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

# Initialize or load the Chroma vector store
vector_store = Chroma(
    collection_name="needlu_docs",
    persist_directory=db_location,
    embedding_function=embeddings
)


if add_documents:
    vector_store.add_documents(documents)
    vector_store.persist()
    print("✅ Documents added to Chroma and saved.")
else:
    print("📦 Loaded existing Chroma DB.")


retriever = vector_store.as_retriever(search_kwargs={"k": 5})


