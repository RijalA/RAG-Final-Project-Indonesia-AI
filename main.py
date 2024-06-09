from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import *

app = FastAPI(
    title='Webservice Final Project Indonesia AI',
    version="1.0.0"
)

app.config = {
    "max_upload_size": "10MB"
}

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(knowledgebase_router)
app.include_router(chat_router)