from pathlib import Path
import os
from functools import cache
import logging

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(Path.cwd() / ".env")


def _setup_logging() -> None:
    """Configure basic logging"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s: %(levelname)s:    %(message)s"
    )


class LLMSettings(BaseModel):
    """Settings for LLM interactions"""
    temperature: float = 0.0
    top_p: float = 0.95


class OpenAISettings(LLMSettings):
    """Settings specific to OpenAI models. Extends LLMSettings."""
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    default_model: str = Field(default="gpt-4o-mini")
    embeddings_model: str = Field(default="text-embedding-3-small")

  
class VectorStoreSettings(BaseModel):
    """Settings for the vector store"""
    service_url: str = Field(default_factory=lambda: os.getenv("TIMESCALE_SERVICE_URL"))
    table_name: str = "documents"
    embedding_dimenstions: int = 1536


class Settings(BaseModel):
    """Main settings class that combines all settings"""
    openai_settings: OpenAISettings = Field(default_factory=OpenAISettings)
    vector_store_settings: VectorStoreSettings = Field(default_factory=VectorStoreSettings)


@cache
def get_settings() -> Settings:
    """Returns all settings and sets up logging"""
    settings = Settings()
    _setup_logging()
    return settings
