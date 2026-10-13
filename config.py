import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Base Project Path
BASE_DIR = Path(__file__).resolve().parent

# Directory Settings
UPLOADS_DIR = BASE_DIR / os.getenv("UPLOADS_DIR", "uploads")
VECTOR_STORE_DIR = BASE_DIR / os.getenv("VECTOR_STORE_DIR", "vector_store")
DATABASE_DIR = BASE_DIR / os.getenv("DATA_DIR", "database")
LOG_DIR = BASE_DIR / os.getenv("LOG_DIR", "logs")

# Ensure required directories exist
for directory in [UPLOADS_DIR, VECTOR_STORE_DIR, DATABASE_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Database File Path
DB_PATH = DATABASE_DIR / "assistant_memory.db"

# LLM & Embedding Settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# Verifier Agent mode: "llm" (default, more thorough - detects real cross-source
# conflicts, costs one extra LLM call) or "deterministic" (rule-based, zero LLM
# calls, faster/free, but only catches "unsupported" answers, not real conflicts).
VERIFIER_MODE = os.getenv("VERIFIER_MODE", "llm")

# Retrieval Confidence Thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.70
MEDIUM_CONFIDENCE_THRESHOLD = 0.40

def get_llm(temperature: float = 0.2):
    """
    Returns an LLM instance based on available configuration, in this priority order:
      1. Groq (GROQ_API_KEY) - free tier available, and by far the fastest option since it
         runs open models on Groq's LPU hardware instead of local CPU/GPU. Recommended default
         for students / free-tier use, since it avoids both cost and local-hardware slowness.
      2. OpenAI (OPENAI_API_KEY) - paid, pay-as-you-go, no free tier by default.
      3. Local Ollama (OLLAMA_MODEL) - always free, but latency depends entirely on your own
         CPU/GPU. Use a small model (e.g. "llama3.2:3b", "phi3:mini") here if you don't have a GPU.
    """
    if GROQ_API_KEY:
        try:
            from langchain_groq import ChatGroq
            preferred_models = [GROQ_MODEL, "openai/gpt-oss-20b", "qwen/qwen3.8-27b", "groq/compound"]
            for model_name in preferred_models:
                try:
                    return ChatGroq(model=model_name, temperature=temperature, api_key=GROQ_API_KEY)
                except Exception:
                    continue
        except Exception:
            pass

    if OPENAI_API_KEY:
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4o-mini", temperature=temperature, api_key=OPENAI_API_KEY)
        except Exception:
            pass

    try:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            temperature=temperature
        )
    except Exception:
        # Fallback to standard generic Ollama call if import varies
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            temperature=temperature
        )
