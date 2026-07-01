import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api.v1.api import api_router
from app.api.v1.endpoints import proxy
from app.core.config import settings, BASE_DIR

app = FastAPI(title="AI Life Legacy Server")

# Ensure required directories exist
GENERATED_PDFS_DIR = BASE_DIR / "generated_pdfs"
STORAGE_DATA_DIR = BASE_DIR / "storage" / "data"
STORAGE_ASSETS_DIR = BASE_DIR / "storage" / "assets"
CACHE_DIR = BASE_DIR / ".cache" / "autobiography"

os.makedirs(GENERATED_PDFS_DIR, exist_ok=True)
os.makedirs(STORAGE_DATA_DIR, exist_ok=True)
os.makedirs(STORAGE_ASSETS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Mount static files
app.mount(
    "/generated-pdfs",
    StaticFiles(directory=str(GENERATED_PDFS_DIR)),
    name="generated_pdfs",
)
app.mount(
    "/storage/data",
    StaticFiles(directory=str(STORAGE_DATA_DIR)),
    name="storage_data",
)
app.mount(
    "/storage/assets",
    StaticFiles(directory=str(STORAGE_ASSETS_DIR)),
    name="storage_assets",
)

# v1 prefix router
app.include_router(api_router, prefix="/api/v1")

# Root level proxy router (for backend compatibility)
app.include_router(proxy.router, tags=["proxy"])

@app.get("/")
def health_check():
    return {"status": "ok", "service": "AI Life Legacy Server"}

@app.on_event("startup")
async def startup_event():
    print("=" * 50)
    print(f"AI Server is running at: http://localhost:{settings.PORT}")
    print(f"Swagger Documentation: http://localhost:{settings.PORT}/docs")
    print(f"OPENAI_API_KEY loaded: {bool(settings.OPENAI_API_KEY)}")
    print(f"OPENAI_PROJECT_ID loaded: {bool(settings.OPENAI_PROJECT_ID)}")
    print(f"OPENAI_SERVICE_ACCOUNT_ID loaded: {bool(settings.OPENAI_SERVICE_ACCOUNT_ID)}")
    print("=" * 50)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=True)
