# day41_rag_basic.py
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── 가상의 런북 데이터 (실제론 docs/ 폴더에서 불러옴) ───
RUNBOOKS = [
    "OOMKilled 발생 시: Pod의 memory limit을 늘리고 메모리 프로파일링을 수행한다.",
    "CrashLoopBackOff 발생 시: 로그를 확인하고 liveness probe 설정을 점검한다.",
    "DB 응답 지연 시: slow query log를 확인하고 인덱스를 점검한다.",
    "디스크 풀 시: 오래된 로그를 정리하고 PVC 용량을 확장한다.",
]

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """두 벡터 사이의 코사인 유사도를 계산한다."""
    # 힌트: dot product / (norm_a * norm_b)
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot / (norm_a * norm_b + 1e-8)  # 분모 0 방지

def fake_embed(text: str) -> np.ndarray:
    """
    실제론 sentence-transformers를 쓰지만,
    오늘은 글자 수 기반 가짜 벡터로 개념만 잡는다.
    """
    np.random.seed(len(text))  # 같은 텍스트 → 같은 벡터
    return np.random.randn(8)  # 차원수: 8

def build_knowledge_base(docs: list[str]) -> tuple[list[np.ndarray], list[str]]:
    """문서 리스트를 벡터로 변환해서 KB를 구축한다."""
    vectors = []
    for doc in docs:
        vec = fake_embed(doc)  # fake_embed 사용
        vectors.append(vec)
        logger.info("임베딩 생성: %s...", doc[:20])
    return vectors, docs

def retrieve(query: str, vectors: list, docs: list, top_k: int = 2) -> list[str]:
    top_k = min(top_k, len(docs))  # ← 방어 코드 추가
    query_vec = fake_embed(query)

    scores = []
    for vec in vectors:
        score = cosine_similarity(vec, query_vec)
        scores.append(score)

    logger.info("검색 쿼리: %s | top_k: %d | 최고점수: %.4f",
                query, top_k, max(scores))  # ← 로깅 추가

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [docs[i] for i in top_indices[:top_k]]

def generate_answer(query: str, context_docs: list[str]) -> str:
    """(가짜) LLM: context를 붙여서 답변을 생성한다."""
    context = "\n".join(context_docs)
    # 실제론 OpenAI / claude API 호출
    return f"[검색된 근거]\n{context}\n\n[답변] 관련 런북을 참고하세요."

if __name__ == "__main__":
    vectors, docs = build_knowledge_base(RUNBOOKS)
    
    query = "Pod가 계속 재시작돼요"
    retrieved = retrieve(query, vectors, docs, top_k=2)
    answer = generate_answer(query, retrieved)
    
    print(answer)