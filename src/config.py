"""Singleton config loaded from config.yaml + .env at import time."""
import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

load_dotenv()

_config_path = Path(__file__).parent.parent / "config" / "config.yaml"
with open(_config_path) as _f:
    _yaml = yaml.safe_load(_f)

cfg = {
    "api_key":        os.getenv("OPENROUTER_API_KEY", ""),
    "cohere_api_key": os.getenv("COHERE_API_KEY", ""),
    "embed_model":    _yaml["embeddings-model"]["name"],
    "llm_model":      _yaml["llm-model"]["name"],
    "reranker_model": _yaml["reranker-model"]["name"],
    "vision_model":   _yaml["vision-model"]["name"],
    "embed_url":      "https://openrouter.ai/api/v1/embeddings",
    "chat_url":       "https://openrouter.ai/api/v1/chat/completions",
}

CHROMA_DIR = str(Path(__file__).parent.parent / "config" / "chroma")
