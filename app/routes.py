# from fastapi import APIRouter # type: ignore
# from pydantic import BaseModel
# from .utils import analyze_news

# router = APIRouter()

# class NewsInput(BaseModel):
#     text: str

# @router.post("/analyze")
# def analyze(news: NewsInput):
#     result = analyze_news(news.text)
#     return result