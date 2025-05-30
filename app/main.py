from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import AnalysisRequest, AnalysisResponse
from .utils import analyze_news

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    # Add your production domains here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_news_endpoint(request: AnalysisRequest):
    try:
        result = analyze_news(request.content)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"status": "API is working!", "docs": "/docs"}
























# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from typing import Optional
# from .schemas import AnalysisRequest, AnalysisResponse
# from .utils import analyze_news

# app = FastAPI()

# # CORS Settings
# origins = [
#     "http://localhost:3000",
#     "http://localhost:8000",
#     # Add your production domains here
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.post("/analyze", response_model=AnalysisResponse)
# async def analyze_news_endpoint(request: AnalysisRequest):
#     try:
#         result = analyze_news(request.content)
#         return result
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/")
# def home():
#     return {"status": "API is working!", "docs": "/docs"}














# ---------------









# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware  # <-- Add this
# from app.routes import router  # Assuming your route handlers are in app/routes.py

# app = FastAPI()

# # 🔐 CORS Settings — Add this block
# origins = [
#     "http://localhost:3000",  # Your Next.js frontend during development
#     # You can add other domains like:
#     # "https://yourdomain.com",
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,         # Use ["*"] for dev if you want to allow everything
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Include your API routes
# app.include_router(router)

# # Optional root endpoint for testing
# @app.get("/")
# def home():
#     return {"status": "API is working!"}






# -------------------


