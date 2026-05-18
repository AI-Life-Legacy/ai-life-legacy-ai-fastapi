from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# 프로젝트 루트: .../ai-life-legacy-ai-fastapi
BASE_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = BASE_DIR / ".env"

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    OPENAI_PROJECT_ID: str | None = None  # 참고용
    OPENAI_SERVICE_ACCOUNT_ID: str | None = None  # 참고용
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_CHAT_MODEL: str = "gpt-4.1-mini"
    OPENAI_QUESTION_MODEL: str = "gpt-4.1-mini"
    OPENAI_EXTRACT_MODEL: str = "gpt-4.1-mini"
    OPENAI_AUTOBIOGRAPHY_MODEL: str = "gpt-4o"
    OPENAI_ORG_ID: str | None = None
    PORT: int = 8000
    ENVIRONMENT: str = "development"
    CHROMA_DB_PATH: str = str(BASE_DIR / "storage" / "chroma_db")
    AI_SERVER_PUBLIC_URL: str = "http://localhost:8000"


    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
