"""Central settings, loaded from environment variables / a .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


class Settings:
    # Any OpenAI-compatible endpoint works: OpenAI, Ollama (local, free), Groq, etc.
    llm_base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    llm_api_key = os.getenv("LLM_API_KEY", "ollama")  # Ollama ignores the key
    llm_model = os.getenv("LLM_MODEL", "llama3.2")

    raw_dir = ROOT / os.getenv("RAW_DIR", "data/raw")
    chroma_dir = ROOT / os.getenv("CHROMA_DIR", "data/chroma")
    collection = os.getenv("COLLECTION", "gmu_docs")

    chunk_strategy = os.getenv("CHUNK_STRATEGY", "structured")  # "structured" or "page" (baseline)
    chunk_size = int(os.getenv("CHUNK_SIZE", "800"))       # characters
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "150"))  # characters
    top_k = int(os.getenv("TOP_K", "4"))


settings = Settings()
