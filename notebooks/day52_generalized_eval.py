# day52_generalized_eval.py
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
from day47_rag_expand import RUNBOOKS, chunk_by_sentence
from day51_hybrid_search import (
    build_bm25, build_faiss_index,
    hybrid_retrieve, BM25_WEIGHT, VECTOR_WEIGHT
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "jhgan/ko-sroberta-multitask"

# ─── 일반화된 평가셋: 시나리오별 다양한 표현 ───
GENERALIZED_EVAL = {
    "OOMKilled": {
        "queries": [
            "메모리가 부족해요",
            "Pod가 OOM으로 죽었어요",
            "메모리 한도를 초과했어요",
        ],
        "expected": "메모리 누수",
    },
    "CrashLoopBackOff": {
        "queries": [
            "Pod가 계속 재시작돼요",
            "Pod가 자꾸 죽어요",
            "컨테이너가 반복 종료돼요",
        ],
        "expected": "CrashLoopBackOff 대응",
    },
    "DB지연": {
        "queries": [
            "DB가 느려요",
            "쿼리 응답이 너무 늦어요",
            "데이터베이스 타임아웃이 발생해요",
        ],
        "expected": "DB 응답 지연 대응",
    },
    "무관쿼리": {
        "queries": [
            "오늘 점심 뭐 먹을까요",
            "날씨가 어때요",
        ],
        "expected": None,  # 무관 쿼리는 뭘 넣어야 할까?
    },
}

def evaluate_by_scenario(
    eval_set: dict,
    chunks: list[str],
    faiss_index,
    bm25: BM25Okapi,
    model,
) -> dict:
    """
    시나리오별로 정확도를 측정한다.
    각 시나리오에서 몇 가지 표현이 정답을 찾는지 확인.
    """
    scenario_results = {}

    for scenario, data in eval_set.items():  # 딕셔너리 순회
        queries  = data["queries"]
        expected = data["expected"]

        correct = 0
        total   = len(queries)  # queries 개수

        for query in queries:
            retrieved = hybrid_retrieve(
                query, chunks, faiss_index, bm25, model
            )
            context = " ".join(retrieved)

            if expected is None:
                # 무관 쿼리 → 결과 없어야 정답
                if "관련 런북을 찾을 수 없습니다" in context:
                    correct += 1
            else:
                if expected in context:
                    correct += 1

        scenario_results[scenario] = {
            "정확도": correct / total,
            "정답수": correct,
            "총질문": total,
        }
        logger.info("시나리오: %s | %d/%d | %.1f%%",
                    scenario, correct, total, correct/total*100)

    return scenario_results

if __name__ == "__main__":
    all_chunks = []
    for doc in RUNBOOKS:
        all_chunks.extend(chunk_by_sentence(doc))

    model      = SentenceTransformer(MODEL_NAME)
    faiss_index = build_faiss_index(all_chunks, model)
    bm25       = build_bm25(all_chunks)

    results = evaluate_by_scenario(
        GENERALIZED_EVAL, all_chunks, faiss_index, bm25, model
    )

    print("\n=== 시나리오별 평가 결과 ===")
    total_correct = 0
    total_queries = 0

    for scenario, result in results.items():
        print(f"{scenario}: {result['정답수']}/{result['총질문']} "
              f"({result['정확도']:.1%})")
        total_correct += result['정답수']  # 정답수
        total_queries += result['총질문']  # 총질문수

    print(f"\n전체 정확도: {total_correct}/{total_queries} "
          f"({total_correct/total_queries:.1%})")