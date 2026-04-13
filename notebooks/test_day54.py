# test_day54.py
import pytest
from unittest.mock import patch, MagicMock
from day53_llm_intent import (
    get_intent, rag_with_intent,
    _intent_cache, SLEEP_SEC
)
from day47_rag_expand import RUNBOOKS, chunk_by_sentence
from day51_hybrid_search import build_bm25, build_faiss_index
from sentence_transformers import SentenceTransformer

# ─── fixture: 무거운 모델 1번만 로드 ───
@pytest.fixture(scope="module")
def rag_setup():
    all_chunks = []
    for doc in RUNBOOKS:
        all_chunks.extend(chunk_by_sentence(doc))
    model       = SentenceTransformer("jhgan/ko-sroberta-multitask")
    faiss_index = build_faiss_index(all_chunks, model)
    bm25        = build_bm25(all_chunks)
    return model, faiss_index, bm25, all_chunks

# ─── 테스트 1: 캐시 동작 확인 ───
def test_cache_hit(monkeypatch):
    """
    캐시에 있는 쿼리는 API를 호출하지 않는다.
    """
    _intent_cache["테스트 쿼리"] = "캐시된 의도"

    # API 호출 함수를 가짜로 대체
    mock_api = MagicMock(return_value=None)  # 절대 호출 안 돼야 함
    monkeypatch.setattr("day53_llm_intent.model_llm", mock_api)

    result = get_intent("테스트 쿼리")
    assert result == "캐시된 의도"           # "캐시된 의도" 반환해야 함
    mock_api.assert_not_called()   # API 호출 0번 확인

    # 테스트 후 캐시 정리
    del _intent_cache["테스트 쿼리"]

# ─── 테스트 2: 무관 쿼리 필터링 ───
def test_unrelated_query(monkeypatch, rag_setup):
    """
    무관 쿼리는 런북을 찾지 않고 바로 반환한다.
    """
    model, faiss_index, bm25, chunks = rag_setup

    # LLM이 "관련 없는 질문" 반환하도록 Mock
    monkeypatch.setattr(
        "day53_llm_intent.get_intent",
        lambda q: "관련 없는 질문"  # "관련 없는 질문" 반환
    )

    result = rag_with_intent(
        "오늘 점심 뭐 먹을까요",
        chunks, faiss_index, bm25, model
    )
    assert "관련 런북을 찾을 수 없습니다" in result["retrieved"][0]

# ─── 테스트 3: 의도 변환 확인 (실제 API 호출) ───
@pytest.mark.parametrize("query,expected_keyword", [
    ("Pod가 자꾸 죽어요",  "CrashLoopBackOff"),
    ("메모리가 부족해요",  "OOMKilled"),
    ("DB가 느려요",       "DB"),
])
def test_intent_mapping(query, expected_keyword):
    """
    LLM이 한국어 쿼리를 올바른 기술용어로 변환한다.
    실제 API 호출 (1회)
    """
    intent = get_intent(query)  # query 넣기
    assert expected_keyword in intent  # intent에 키워드 있는지

# ─── 테스트 4: 전체 파이프라인 통합 테스트 ───
def test_pipeline_integration(rag_setup):
    """
    전체 파이프라인이 정상 동작하는지 확인.
    """
    model, faiss_index, bm25, chunks = rag_setup

    result = rag_with_intent(
        "Pod가 자꾸 죽어요",
        chunks, faiss_index, bm25, model
    )

    assert result["intent"] is not None     # None이 아니어야 함
    assert len(result["retrieved"]) > 1  # 결과가 1개 이상
    assert "CrashLoopBackOff" in " ".join(result["retrieved"])