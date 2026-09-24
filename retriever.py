import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_retriever(pdf_path: str, k: int = 4):
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    if not pages:
        raise ValueError("The uploaded PDF contains no readable text.")

    # Split document into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(pages)

    if not chunks:
        raise ValueError("Could not create text chunks from the PDF.")

    # Create embeddings (calls HuggingFace's hosted API instead of loading
    # the model locally, so it doesn't blow past Render's free-tier RAM limit)
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=os.environ["HF_API_TOKEN"],
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create FAISS vector store
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    # Return retriever
    return vectorstore.as_retriever(
        search_kwargs={"k": k}
    )
