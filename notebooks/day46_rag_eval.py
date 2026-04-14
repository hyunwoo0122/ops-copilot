# day46_rag_eval.py
import logging
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
THRESHOLD = 0.3

RUNBOOKS = [
    """OOMKilled 발생 시 대응 가이드:
    1단계: kubectl describe pod 명령으로 메모리 사용량을 확인한다.
    2단계: Pod의 memory limit을 현재값의 1.5배로 늘린다.""",
    """CrashLoopBackOff 발생 시 대응 가이드:
    1단계: kubectl logs 명령으로 에러 로그를 확인한다.
    2단계: liveness probe 설정이 올바른지 점검한다.""",
    """DB 응답 지연 발생 시 대응 가이드:
    1단계: slow query log를 확인하고 병목 쿼리를 찾는다.
    2단계: 해당 테이블의 인덱스 적용 여부를 점검한다.""",
]

# ─── 평가셋 (query, expected_keyword) ───
EVAL_SET = [
    ("Pod가 계속 재시작돼요",     "CrashLoopBackOff"),
    ("메모리가 부족해요",         "OOMKilled"),
    ("DB가 느려요",              "DB 응답 지연"),
    ("오늘 점심 뭐 먹을까요",     None),  # 관련 없는 쿼리 → None
]

def chunk_by_sentence(text: str) -> list[str]:
    return [line.strip() for line in text.split('\n') if line.strip()]

def build_faiss_index(docs: list[str], model) -> tuple:
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_by_sentence(doc))
    vectors = np.array(model.encode(all_chunks), dtype=np.float32)
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])  # 어제 배운 것!
    index.add(vectors)
    return index, all_chunks

def retrieve(query, index, chunks, model, top_k=3, threshold=THRESHOLD):
    top_k = min(top_k, len(chunks))
    query_vec = np.array(model.encode([query]), dtype=np.float32)
    faiss.normalize_L2(query_vec)
    distances, indices = index.search(query_vec, top_k)
    results = []
    for score, idx in zip(distances[0], indices[0]):
        if score >= threshold:  # threshold 필터링
            results.append(chunks[idx])
    return results

def evaluate(eval_set: list, index, chunks, model) -> dict:
    """
    평가셋을 돌려서 정확도와 근거율을 계산한다.
    """
    correct = 0   # 정답 키워드가 검색된 횟수
    faithful = 0  # 근거 문서가 있는 횟수
    total = len(eval_set)   # 전체 질문 수

    for query, expected in eval_set:
        retrieved = retrieve(query, index, chunks, model)
        context = " ".join(retrieved)  # 검색된 문서를 하나로 합치기

        # 근거율: 검색 결과가 있으면 근거 있음
        if retrieved:
            faithful += 1

        # 정확도: expected 키워드가 context에 있는지 확인
        if expected is None:
            # 관련 없는 쿼리 → 검색 결과가 없어야 정답
            if not retrieved:  # retrieved가 비어있으면
                correct += 1
        else:
            if expected in context:  # context에 키워드가 있으면
                correct += 1

        logger.info("쿼리: %s | 정답: %s | 검색결과: %s",
                    query, expected, retrieved)

    return {
        "정확도": correct / total,
        "근거율": faithful / total,
        "총 질문수": total,
    }

if __name__ == "__main__":
    model = SentenceTransformer(MODEL_NAME)
    index, chunks = build_faiss_index(RUNBOOKS, model)
    
    result = evaluate(EVAL_SET, index, chunks, model)
    print("\n=== RAG 평가 결과 ===")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2%}")  # 퍼센트로 출력
        else:
            print(f"{key}: {value}")