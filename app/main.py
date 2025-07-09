from fastapi import FastAPI
from app.routers import dataRouter

app = FastAPI()

app.include_router(dataRouter.app)