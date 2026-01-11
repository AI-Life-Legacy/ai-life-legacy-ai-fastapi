from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class RagSyncRequest(BaseModel):
    userId: str
    text: str
    metadata: Dict[str, Any]

class RagSearchRequest(BaseModel):
    userId: str
    query: str

class RagSearchResult(BaseModel):
    text: str
    score: Optional[float]

class RagSearchResponse(BaseModel):
    results: List[RagSearchResult]
