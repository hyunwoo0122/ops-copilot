# test_day48.py
import pytest
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from day47_rag_expand import (
    chunk_by_sentence,
    build_faiss_index,
    retrieve,
    evaluate,
    RUNBOOKS,
    EVAL_SET,
    THRESHOLD,
)

# ─── fixture: 모델/인덱스를 한 번만 로드 ───
@pytest.fixture(scope="module")
def rag_setup():
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    index, chunks = build_faiss_index(RUNBOOKS, model)
    return model, index, chunks

# ─── 테스트 1: 정상 청킹 ───
def test_chunk_basic():
    text = "1단계: 확인\n2단계: 점검\n3단계: 조치"
    result = chunk_by_sentence(text)
    assert len(result) == 3  # 몇 개 나와야 할까?

# ─── 테스트 2: 빈 문자열 ───
def test_chunk_empty():
    result = chunk_by_sentence("")
    assert result == []

# ─── 테스트 3: 빈 줄 제거 ───
def test_chunk_no_blank():
    text = "1단계: 확인\n\n\n2단계: 점검"
    result = chunk_by_sentence(text)
    assert "" not in result  # 빈 문자열이 포함되면 안 됨

# ─── 테스트 4: parametrize로 여러 쿼리 한번에 ───
@pytest.mark.parametrize("query,expected", [
    ("메모리가 부족해요",    "메모리 누수"),
    ("DB가 느려요",         "DB 응답 지연 대응"),
    ("CPU 사용률이 너무 높아요", "CPU 과부하 대응"),
    ("배포가 실패했어요",    "배포 실패 대응"),
])
def test_retrieve_related(query, expected, rag_setup):
    model, index, chunks = rag_setup
    retrieved = retrieve(query, index, chunks, model)
    context = " ".join(retrieved)  # retrieved를 문자열로 합치기
    assert expected in context   # context에 키워드가 있는지

# ─── 테스트 5: 무관 쿼리 필터링 ───
def test_retrieve_unrelated(rag_setup):
    model, index, chunks = rag_setup
    retrieved = retrieve("오늘 점심 뭐 먹을까요",
                         index, chunks, model,
                         threshold=0.5)  # 높은 threshold = 0.5
    assert retrieved == []  # 빈 리스트여야 함

# ─── 테스트 6: 전체 정확도 회귀 테스트 ───
def test_evaluate_accuracy(rag_setup):
    model, index, chunks = rag_setup
    result = evaluate(EVAL_SET, index, chunks, model)
    assert result["정확도"] >= THRESHOLD  # 0.5 이상이어야 함 (회귀 방지)