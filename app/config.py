# Load environment variables
from dotenv import load_dotenv
import os

def load_config():
    load_dotenv()
    
    groq = os.getenv("GROQ_API_KEY")
    pinecone = os.getenv("PINECONE_API_KEY")
    print(f"PINECONE_API_KEY:", "**********{pinecone}" if pinecone else "**********Missing")

    if not groq:
        raise ValueError("GROQ_API_KEY is missing")
    if not pinecone:
        raise ValueError("PINECONE_API_KEY is missing")

    return {
        "GROQ_API_KEY": groq,
        "PINECONE_API_KEY": pinecone
    }