from fastapi import FastAPI
from app.api.v1.api import api_router
from app.api.v1.endpoints import proxy
from app.core.config import settings

app = FastAPI(title="AI Life Legacy Server")

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
    print("=" * 50)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=True)
