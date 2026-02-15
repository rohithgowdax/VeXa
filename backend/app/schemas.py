from pydantic import BaseModel, Field


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="German input text")


class TranslateResponse(BaseModel):
    translation: str
