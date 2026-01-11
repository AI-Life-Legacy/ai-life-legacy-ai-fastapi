from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# 프로젝트 루트: .../ai-life-legacy-ai-fastapi
BASE_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = BASE_DIR / ".env"

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    OPENAI_ORG_ID: str | None = None
    PORT: int = 8000
    ENVIRONMENT: str = "development"
    CHROMA_DB_PATH: str = str(BASE_DIR / "storage" / "chroma_db")

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
