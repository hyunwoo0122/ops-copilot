# day51_hybrid_search.py
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
from notebooks.day47_rag_expand import RUNBOOKS, EVAL_SET, chunk_by_sentence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "jhgan/ko-sroberta-multitask"
THRESHOLD = 0.1
BM25_WEIGHT = 0.1   # BM25 가중치
VECTOR_WEIGHT = 0.9  # 벡터 가중치

def build_bm25(chunks: list[str]) -> BM25Okapi:
    """
    청크를 단어 단위로 쪼개서 BM25 인덱스를 구축한다.
    """
    tokenized = [chunk.split() for chunk in chunks]
    #                   ↑ 공백 기준으로 단어 분리하는 메서드
    return BM25Okapi(tokenized)  # tokenized 넣기

def build_faiss_index(chunks: list[str], model) -> faiss.Index:
    vectors = np.array(model.encode(chunks), dtype=np.float32)
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index

def hybrid_retrieve(
    query: str,
    chunks: list[str],
    faiss_index,
    bm25: BM25Okapi,
    model,
    top_k: int = 3,
    threshold: float = THRESHOLD,
) -> list[str]:
    """벡터 검색 + BM25 점수를 합산해서 검색한다."""

    # ─── 벡터 검색 점수 ───
    query_vec = np.array(model.encode([query]), dtype=np.float32)
    faiss.normalize_L2(query_vec)
    distances, indices = faiss_index.search(query_vec, len(chunks))

    vector_scores = np.zeros(len(chunks))
    for score, idx in zip(distances[0], indices[0]):
        vector_scores[idx] = score

    # ─── BM25 점수 ───
    tokenized_query = query.split()  # 공백 기준 분리
    bm25_scores = bm25.get_scores(tokenized_query)  # tokenized_query 넣기

    # 정규화 (0~1 범위로)
    if bm25_scores.max() > 0:
        bm25_scores = bm25_scores / bm25_scores.max()

    # ─── 하이브리드 점수 합산 ───
    hybrid_scores = (VECTOR_WEIGHT * vector_scores +
                     BM25_WEIGHT * bm25_scores)  # BM25_WEIGHT 사용

    # ─── 상위 top_k 반환 ───
    top_indices = sorted(range(len(hybrid_scores)),
                         key=lambda i: hybrid_scores[i],
                         reverse=True)[:top_k]

    results = []
    for idx in top_indices:
        if hybrid_scores[idx] >= threshold:
            results.append(chunks[idx])
            logger.info("채택: %.4f | %s...", hybrid_scores[idx], chunks[idx][:30])

    return results if results else ["관련 런북을 찾을 수 없습니다."]

def evaluate(eval_set, chunks, faiss_index, bm25, model) -> dict:
    correct = 0
    faithful = 0
    total = len(eval_set)

    for query, expected in eval_set:
        retrieved = hybrid_retrieve(query, chunks, faiss_index, bm25, model)
        context = " ".join(retrieved)

        if retrieved and retrieved[0] != "관련 런북을 찾을 수 없습니다.":
            faithful += 1

        if expected is None:
            if not retrieved or retrieved[0] == "관련 런북을 찾을 수 없습니다.":
                correct += 1
        else:
            if expected in context:  # context에 키워드 있는지
                correct += 1

        logger.info("쿼리: %s | 정답: %s | 결과: %s",
                    query, expected, retrieved)

    return {
        "정확도": correct / total,
        "근거율": faithful / total,
        "총 질문수": total,
    }

if __name__ == "__main__":
    # 청크 준비
    all_chunks = []
    for doc in RUNBOOKS:
        all_chunks.extend(chunk_by_sentence(doc))

    # 인덱스 구축
    model = SentenceTransformer(MODEL_NAME)
    faiss_index = build_faiss_index(all_chunks, model)
    bm25 = build_bm25(all_chunks)  # all_chunks 넣기

    # 평가
    result = evaluate(EVAL_SET, all_chunks, faiss_index, bm25, model)

    print("\n=== 하이브리드 검색 평가 결과 ===")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2%}")
        else:
            print(f"{key}: {value}")