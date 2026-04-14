# day58_fastapi_async.py
import os
import time
import asyncio
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging

from notebooks.day47_rag_expand import RUNBOOKS, chunk_by_sentence
from notebooks.day51_hybrid_search import build_bm25, build_faiss_index, hybrid_retrieve
from notebooks.day53_llm_intent import get_intent
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OpsCopilot API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start    = time.time()
    response = await call_next(request)  # await 추가
    elapsed  = time.time() - start
    logger.info("%s %s → %d | %.3fs",
                request.method, request.url.path,
                response.status_code, elapsed)
    return response

MODEL_NAME  = "jhgan/ko-sroberta-multitask"
embed_model = SentenceTransformer(MODEL_NAME)
all_chunks  = []
for doc in RUNBOOKS:
    all_chunks.extend(chunk_by_sentence(doc))
faiss_index = build_faiss_index(all_chunks, embed_model)
bm25        = build_bm25(all_chunks)

class AnalyzeRequest(BaseModel):
    query: str
    top_k: int = 2

@app.get("/health")
async def health_check():  # async 추가
    return {"status": "ok", "version": "0.3.0", "chunks": len(all_chunks)}

@app.get("/analyze")
async def analyze_get(query: str):  # async 추가
    if not query.strip():
        raise HTTPException(status_code=400, detail="query가 비어있습니다")
    try:
        # LLM 호출을 비동기로 처리
        intent = await asyncio.to_thread(get_intent, query)  # await 추가
        retrieved = hybrid_retrieve(
            intent, all_chunks, faiss_index, bm25, embed_model
        )
        return {"query": query, "intent": intent, "retrieved": retrieved}
    except Exception as e:
        logger.error("분석 실패: %s", e)
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")

@app.post("/analyze")
async def analyze_post(request: AnalyzeRequest):  # async 추가
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query가 비어있습니다")
    try:
        intent = await asyncio.to_thread(get_intent, request.query)  # await
        retrieved = hybrid_retrieve(
            intent, all_chunks, faiss_index, bm25, embed_model,
            top_k=request.top_k
        )
        return {"query": request.query, "intent": intent, "retrieved": retrieved}
    except Exception as e:
        logger.error("분석 실패: %s", e)
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("day58_fastapi_async:app",
                host="0.0.0.0", port=8000, reload=True)