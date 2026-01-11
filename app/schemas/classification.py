from pydantic import BaseModel

class UserCaseRequest(BaseModel):
    introText: str

class UserCaseResponse(BaseModel):
    case: str
    reasoning: str | None = None
