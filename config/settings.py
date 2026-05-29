from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    DEFAULT_MODEL: Literal["bart", "t5"] = "bart"
    DEFAULT_METHOD: Literal["extractive", "abstractive"] = "extractive"
    DEFAULT_LENGTH: Literal["short", "medium", "long"] = "medium"
    MIN_TEXT_LENGTH: int = 50
    MAX_TEXT_LENGTH: int = 10000
    CACHE_SIZE: int = 100
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False

    class Config:
        env_file = ".env"

settings = Settings()