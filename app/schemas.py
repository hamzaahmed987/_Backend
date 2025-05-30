from pydantic import BaseModel
from typing import Optional, List

class AnalysisRequest(BaseModel):
    content: str

class AnalysisResponse(BaseModel):
    is_fake: bool
    confidence: float
    analysis: str
    sources: List[str] = []
    public_sentiment: Optional[str] = None
    detected_language: Optional[str] = None
    sample_tweets: Optional[List[str]] = None