import os
import time
import logging
import httpx
from dotenv import load_dotenv
from openai import OpenAI

# Load from .env file if it exists
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "none")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/invoice_insight")

def get_llm_client() -> OpenAI:
    """Return an OpenAI client configured for the specified API endpoint using explicit httpx.Client."""
    http_client = httpx.Client(follow_redirects=True)
    return OpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY if LLM_API_KEY not in ("none", "") else "placeholder",
        http_client=http_client
    )

def llm_chat_completion(client: OpenAI, messages: list, model: str = None, temperature: float = 0, max_tokens: int = 1000, max_retries: int = 3):
    """Wrapper around OpenAI chat completion with retry and exponential backoff on rate limit errors."""
    _model = model or LLM_MODEL
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response
        except Exception as e:
            error_str = str(e)
            if ("429" in error_str or "rate limit" in error_str.lower()) and attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 5  # 5s, 10s, 20s
                logger.warning(f"Rate limited (429). Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            elif attempt < max_retries - 1 and ("timeout" in error_str.lower() or "connection" in error_str.lower()):
                wait_time = 3
                logger.warning(f"Network error: {error_str}. Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise
