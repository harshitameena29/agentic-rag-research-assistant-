import os
from typing import List

from huggingface_hub import InferenceClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class HFInferenceEmbeddings(Embeddings):
    """Embeddings via HuggingFace's hosted Inference API (router.huggingface.co),
    so no model is loaded into local RAM. Replaces the deprecated
    langchain_community HuggingFaceInferenceAPIEmbeddings, which still points
    at the retired api-inference.huggingface.co endpoint."""

    def __init__(self, model_name: str, api_key: str):
        self._client = InferenceClient(model=model_name, token=api_key)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self._client.feature_extraction(texts)
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


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
    embeddings = HFInferenceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        api_key=os.environ["HF_API_TOKEN"]
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
