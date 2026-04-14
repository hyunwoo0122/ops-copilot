# day53_llm_intent.py
import os
from dotenv import load_dotenv
load_dotenv()  # ← 가장 먼저 실행

import logging
import time
import google.generativeai as genai
from day47_rag_expand import RUNBOOKS, chunk_by_sentence
from day51_hybrid_search import (
    build_bm25, build_faiss_index, hybrid_retrieve
)
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME   = "jhgan/ko-sroberta-multitask"
GEMINI_MODEL = "models/gemini-flash-lite-latest"
SLEEP_SEC    = 30  # 매직 넘버 금지 → 상수로 관리

# ─── Gemini 클라이언트 초기화 ───
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

SYSTEM_PROMPT = """
너는 쿠버네티스/클라우드 운영 전문가야.
사용자의 질문을 보고 어떤 기술적 문제인지
전문 용어로 한 줄만 답해줘.
예시:
- "Pod가 자꾸 죽어요" → "CrashLoopBackOff 문제"
- "메모리가 부족해요" → "OOMKilled 문제"
- "DB가 느려요"       → "DB 응답 지연 문제"
관련 없는 질문이면 "관련 없는 질문" 이라고만 답해줘.
"""

# ─── 최적화 1: 캐시 ───
_intent_cache: dict[str, str] = {}

def get_intent(query: str, max_retry: int = 3) -> str:
    """
    LLM으로 사용자 의도를 파악한다.
    - 캐싱: 같은 쿼리는 API 안 부름
    - 지수 백오프: 429 에러 시 자동 재시도
    - 폴백: 최종 실패 시 원본 쿼리 반환
    """
    # ─── 최적화 1: 캐시 확인 ───
    if query in _intent_cache:
        logger.info("캐시 히트 → API 호출 생략: %s", query)
        return _intent_cache[query]

    logger.info("의도 파악 시작: %s", query)

    model_llm = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT
    )

    # ─── 최적화 2: 지수 백오프 ───
    for attempt in range(max_retry):
        try:
            response = model_llm.generate_content(query)
            intent   = response.text.strip()
            logger.info("파악된 의도: %s", intent)

            _intent_cache[query] = intent  # 캐시 저장
            return intent

        except Exception as e:
            if "429" in str(e):
                wait = SLEEP_SEC * (2 ** attempt)  # 30 → 60 → 120초
                logger.warning("할당량 초과. %d초 대기 (시도 %d/%d)",
                               wait, attempt + 1, max_retry)
                time.sleep(wait)
            else:
                logger.error("LLM 호출 실패: %s", e)
                break  # 429 아닌 에러는 재시도 안 함

    # ─── 폴백 ───
    logger.warning("폴백 실행: 원본 쿼리로 검색")
    return query

def rag_with_intent(
    query: str,
    chunks: list[str],
    faiss_index,
    bm25,
    model,
) -> dict:
    """
    1. LLM으로 의도 파악
    2. 의도로 런북 검색
    3. 결과 반환
    """
    intent = get_intent(query)

    if "관련 없는 질문" in intent:
        return {
            "query":     query,
            "intent":    intent,
            "retrieved": ["관련 런북을 찾을 수 없습니다."],
        }

    retrieved = hybrid_retrieve(
        intent, chunks, faiss_index, bm25, model
    )

    return {
        "query":     query,
        "intent":    intent,
        "retrieved": retrieved,
    }

if __name__ == "__main__":
    # 준비
    all_chunks = []
    for doc in RUNBOOKS:
        all_chunks.extend(chunk_by_sentence(doc))

    embed_model = SentenceTransformer(MODEL_NAME)
    faiss_index = build_faiss_index(all_chunks, embed_model)
    bm25        = build_bm25(all_chunks)

    test_queries = [
        "Pod가 자꾸 죽어요",
        "메모리가 부족해요",
        "DB가 느려요",
        "오늘 점심 뭐 먹을까요",
        "Pod가 자꾸 죽어요",  # ← 캐시 테스트용 중복 쿼리
    ]

    for i, query in enumerate(test_queries):
        result = rag_with_intent(
            query, all_chunks, faiss_index, bm25, embed_model
        )
        print(f"\n질문: {result['query']}")
        print(f"의도: {result['intent']}")
        print(f"검색결과: {result['retrieved'][0][:50]}...")

        # ─── 최적화 3: 마지막 쿼리는 대기 안 함 ───
        # ─── 최적화 4: 캐시 히트면 대기 안 함 ───
        is_last    = (i == len(test_queries) - 1)
        is_cached  = (query in _intent_cache)

        if not is_last and not is_cached:
            logger.info("다음 요청 전 %d초 대기...", SLEEP_SEC)
            time.sleep(SLEEP_SEC)