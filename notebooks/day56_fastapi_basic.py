# day56_fastapi_basic.py
import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging

from day47_rag_expand import RUNBOOKS, chunk_by_sentence
from day51_hybrid_search import build_bm25, build_faiss_index, hybrid_retrieve
from day53_llm_intent import get_intent
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── 앱 생성 ───
app = FastAPI(
    title="OpsCopilot API",
    description="로그 이상 분석 + 런북 검색 API",
    version="0.1.0",
)

# ─── 모델/인덱스 초기화 (앱 시작 시 1번만) ───
MODEL_NAME = "jhgan/ko-sroberta-multitask"
embed_model = SentenceTransformer(MODEL_NAME)

all_chunks = []
for doc in RUNBOOKS:
    all_chunks.extend(chunk_by_sentence(doc))

faiss_index = build_faiss_index(all_chunks, embed_model)
bm25        = build_bm25(all_chunks)

# ─── Request Body 스키마 ───
class AnalyzeRequest(BaseModel):
    query: str   # 문자열 타입
    top_k: int = 2

# ─── 엔드포인트 1: 헬스체크 ───
@app.get("/health")  # GET 메서드
def health_check():
    return {"status": "ok", "version": "0.1.0"}

# ─── 엔드포인트 2: GET 분석 ───
@app.get("/analyze")
def analyze_get(query: str):  # URL 파라미터
    """GET /analyze?query=Pod가 죽어요"""
    if not query:
        raise HTTPException(status_code=400, detail="query가 비어있습니다")

    intent    = get_intent(query)
    retrieved = hybrid_retrieve(
        intent, all_chunks, faiss_index, bm25, embed_model
    )
    return {
        "query":     query,
        "intent":    intent,
        "retrieved": retrieved,
    }

# ─── 엔드포인트 3: POST 분석 ───
@app.post("/analyze")
def analyze_post(request: AnalyzeRequest):  # Request Body
    """POST /analyze {"query": "Pod가 죽어요"}"""
    intent    = get_intent(request.query)  # query 꺼내기
    retrieved = hybrid_retrieve(
        intent, all_chunks, faiss_index, bm25, embed_model,
        top_k=request.top_k  # top_k 꺼내기
    )
    return {
        "query":     request.query,
        "intent":    intent,
        "retrieved": retrieved,
    }

if __name__ == "__main__":
    uvicorn.run(
        "day56_fastapi_basic:app",
        host="0.0.0.0",
        port=8000,   # 기본 포트
        reload=True
    )