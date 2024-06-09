from fastapi import APIRouter, HTTPException, Form, UploadFile, status
from typing import List

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import utils as chromautils

from config.constant import embedding

import os
import uuid

router = APIRouter(
    prefix="/v1/knowledgebase",
    tags=["Knowledge Base"],
)

@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload(files: List[UploadFile]):
    path_kb = "./kb"
    loaders = []
    
    for file in files:
        save_path = f"{path_kb}/{file.filename}"
        
        if file.content_type == "text/csv":
            contents = await file.read()

            with open(save_path, "wb") as f:
                f.write(contents)

            loaders.append(CSVLoader(save_path))
        else:
            raise HTTPException(status_code=400, detail="Invalid file type")

    docs = []
    for loader in loaders:
        docs.extend(chromautils.filter_complex_metadata(loader.load()))
    
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size = 1000,
    #     chunk_overlap = 100
    # )
    # splits = text_splitter.split_documents(docs)
    ids = [str(uuid.uuid1()) for i in range(len(docs))]

    path_chroma = f"{path_kb}/chroma"
    if not os.path.exists(path_chroma):
        vector_db = Chroma.from_documents(
            documents=docs,
            embedding=embedding,
            persist_directory=path_chroma,
            ids=ids
        )
    else:
        vector_db = Chroma(
            embedding_function=embedding,
            persist_directory=path_chroma,
        )
        vector_db.add_documents(
            documents=docs,
            embedding=embedding,
            persist_directory=path_chroma,
            ids=ids
        )
    
    return "File uploaded successfully"