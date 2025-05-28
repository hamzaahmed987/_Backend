from fastapi import FastAPI
from app.routes import router

app = FastAPI()
app.include_router(router)

@app.get("/")
def home():
    return {"status": "API is working!"}





# ---------------------














# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from app.routes import router

# app = FastAPI(title="News Verification API", version="1.0.0")

# # CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Include all routes
# app.include_router(router)

# @app.get("/")
# async def root():
#     return {"message": "News Verification API is running"}

# @app.get("/health")
# async def health_check():
#     return {"status": "healthy"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
