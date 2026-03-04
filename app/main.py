# main.py
import os
import requests
import warnings
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from config import load_config
from utils import extract_text_from_pdf, chunk_text
from embeddings import create_embeddings, init_pinecone, upsert_chunks
from rag import ask_rag
import uvicorn

warnings.filterwarnings("ignore")

# Global state
state = {}

# Lifespan for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()

    url = "https://alexjungaalto.github.io/MLBasicsBook.pdf"
    pdf_path = "MLBasicsBook.pdf"

    # Download PDF if not exists
    if not os.path.exists(pdf_path):
        response = requests.get(url)
        with open(pdf_path, "wb") as f:
            f.write(response.content)
        print("PDF downloaded successfully!")
    else:
        print("Using cached PDF:", pdf_path)

    # Extract text and chunk it
    document_text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(document_text)

    # Apply sample limit if set in env
    sample_limit = os.getenv("SAMPLE_LIMIT")
    if sample_limit:
        try:
            chunks = chunks[:int(sample_limit)]
        except ValueError:
            pass

    # Create embeddings and Pinecone index
    chunk_embeddings, embedding_model = create_embeddings(chunks)
    index = init_pinecone(config.get("PINECONE_API_KEY"), "ml-basics-book")
    upsert_chunks(index, chunks, chunk_embeddings)

    # Store global state
    state["client"] = Groq(api_key=config.get("GROQ_API_KEY"))
    state["index"] = index
    state["embedding_model"] = embedding_model

    print("Startup complete!")
    yield  # App runs here
    state.clear()  # Cleanup on shutdown

# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# Pydantic model for request body
class QuestionRequest(BaseModel):
    question: str

# Health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

# Ask endpoint
@app.post("/ask")
def ask(request: QuestionRequest):
    if not state:
        raise HTTPException(status_code=503, detail="App not ready")
    answer = ask_rag(
        request.question,
        state["client"],
        state["index"],
        state["embedding_model"]
    )
    return {"question": request.question, "answer": answer}

# Uvicorn entry point
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True
    )