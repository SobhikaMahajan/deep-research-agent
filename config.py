"""
Centralized configuration module using Python dataclasses.
Supports Gemini (primary) and Groq (fallback) LLMs, with ChromaDB for search.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class GeminiConfig:
    """Configuration for Gemini API (Google AI Studio, OpenAI-compatible)."""
    api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    base_url: str = field(default_factory=lambda: os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"))
    temperature: float = 0.0

    def validate(self) -> bool:
        """Check all required fields are set."""
        return bool(self.api_key and self.model and self.base_url)


@dataclass
class GroqConfig:
    """Configuration for Groq API (fallback, higher free-tier rate limits)."""
    api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    base_url: str = field(default_factory=lambda: os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"))
    temperature: float = 0.0

    def validate(self) -> bool:
        """Check all required fields are set."""
        return bool(self.api_key and self.model and self.base_url)


@dataclass
class ChromaDBConfig:
    """Configuration for ChromaDB local vector search."""
    persist_dir: str = field(default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
    collection_name: str = field(default_factory=lambda: os.getenv("CHROMA_COLLECTION_NAME", "internal_documents"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    default_top_k: int = 5

    def validate(self) -> bool:
        """Check all required fields are set."""
        return bool(self.persist_dir and self.collection_name and self.embedding_model)


@dataclass
class LLMEndpointConfig:
    """LLM endpoint configuration compatible with AG2."""
    model: str = ""
    base_url: str = ""
    api_type: str = "openai"
    api_key: Optional[str] = None
    temperature: float = 0.0

    def validate(self) -> bool:
        """Check all required fields are set."""
        return bool(self.model and self.base_url and self.api_key)


class AppConfig:
    """Main application configuration."""

    def __init__(self):
        self.gemini = GeminiConfig()
        self.groq = GroqConfig()
        self.chroma = ChromaDBConfig()

    def create_llm_config(self) -> LLMEndpointConfig:
        """Create primary LLM endpoint config from Gemini settings."""
        return LLMEndpointConfig(
            model=self.gemini.model,
            base_url=self.gemini.base_url,
            api_type="openai",
            api_key=self.gemini.api_key,
            temperature=self.gemini.temperature,
        )

    def create_fallback_llm_config(self) -> Optional[LLMEndpointConfig]:
        """Create fallback LLM endpoint config from Groq settings."""
        if not self.groq.validate():
            return None
        return LLMEndpointConfig(
            model=self.groq.model,
            base_url=self.groq.base_url,
            api_type="openai",
            api_key=self.groq.api_key,
            temperature=self.groq.temperature,
        )

    def validate_all(self) -> dict:
        """Validate all configuration sections."""
        return {
            "gemini": self.gemini.validate(),
            "groq": self.groq.validate(),
            "chroma": self.chroma.validate(),
        }


# Global instance
config = AppConfig()
