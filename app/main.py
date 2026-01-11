from fastapi import FastAPI
from app.api.v1.api import api_router
from app.core.config import settings

app = FastAPI(title="AI Life Legacy Server")

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def health_check():
    return {"status": "ok", "service": "AI Life Legacy Server"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=True)
