# day57_fastapi_middleware.py
import os
import time
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging

from day47_rag_expand import RUNBOOKS, chunk_by_sentence
from day51_hybrid_search import build_bm25, build_faiss_index, hybrid_retrieve
from day53_llm_intent import get_intent
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="OpsCopilot API",
    description="로그 이상 분석 + 런북 검색 API",
    version="0.2.0",
)

# ─── CORS 미들웨어 ───
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # 모든 도메인 허용
    allow_methods=["*"],     # 모든 HTTP 메서드 허용
    allow_headers=["*"],
)

# ─── 요청 시간 측정 미들웨어 ───
@app.middleware("http") # middleware, "http"
async def log_requests(request: Request, call_next):
    start  = time.time()
    response = await call_next(request)  # 실제 API 처리
    elapsed  = time.time() - start

    logger.info(
        "%s %s → %d | %.3fs",
        request.method,   # GET / POST
        request.url.path, # /analyze
        response.status_code,
        elapsed,
    )
    return response  # response 반환

# ─── 모델 초기화 ───
MODEL_NAME  = "jhgan/ko-sroberta-multitask"
embed_model = SentenceTransformer(MODEL_NAME)

all_chunks  = []
for doc in RUNBOOKS:
    all_chunks.extend(chunk_by_sentence(doc))

faiss_index = build_faiss_index(all_chunks, embed_model)
bm25        = build_bm25(all_chunks)

# ─── 스키마 ───
class AnalyzeRequest(BaseModel):
    query: str
    top_k: int = 2

# ─── 헬스체크 ───
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "version": "0.2.0",
        "chunks":  len(all_chunks),  # 청크 수 반환
    }

# ─── GET 분석 ───
@app.get("/analyze")
def analyze_get(query: str):
    if not query.strip():  # 공백만 있는 경우도 처리
        raise HTTPException(
            status_code=400,  # 400
            detail="query가 비어있습니다"
        )
    try:
        intent    = get_intent(query)
        retrieved = hybrid_retrieve(
            intent, all_chunks, faiss_index, bm25, embed_model
        )
        return {"query": query, "intent": intent, "retrieved": retrieved}

    except Exception as e:
        logger.error("분석 실패: %s", e)
        raise HTTPException(
            status_code=500,  # 500
            detail=f"서버 오류: {str(e)}"
        )

# ─── POST 분석 ───
@app.post("/analyze")
def analyze_post(request: AnalyzeRequest):
    if not request.query.strip():  # 공백 제거 후 빈 문자열 확인
        raise HTTPException(status_code=400, detail="query가 비어있습니다")
    try:
        intent    = get_intent(request.query)
        retrieved = hybrid_retrieve(
            intent, all_chunks, faiss_index, bm25, embed_model,
            top_k=request.top_k
        )
        return {
            "query":     request.query,
            "intent":    intent,
            "retrieved": retrieved,
        }
    except Exception as e:
        logger.error("분석 실패: %s", e)
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "day57_fastapi_middleware:app",
        host="0.0.0.0",   # "0.0.0.0"
        port=8000,
        reload=True
    )