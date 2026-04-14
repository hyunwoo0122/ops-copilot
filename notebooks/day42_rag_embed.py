# day42_rag_embed.py
import numpy as np
import logging
from sentence_transformers import SentenceTransformer  # ← 어떤 클래스를 import?
import faiss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RUNBOOKS = [
    "OOMKilled 발생 시: Pod의 memory limit을 늘리고 메모리 프로파일링을 수행한다.",
    "CrashLoopBackOff 발생 시: 로그를 확인하고 liveness probe 설정을 점검한다.",
    "DB 응답 지연 시: slow query log를 확인하고 인덱스를 점검한다.",
    "디스크 풀 시: 오래된 로그를 정리하고 PVC 용량을 확장한다.",
]

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  

def build_faiss_index(docs: list[str], model) -> tuple:
    """문서를 임베딩해서 FAISS 인덱스를 구축한다."""
    logger.info("임베딩 시작: 총 %d개 문서", len(docs))
    
    # 문서 전체를 한번에 벡터로 변환
    vectors = model.encode(docs)  # ← .encode()? .embed()? Doc 힌트 참고
    vectors = np.array(vectors, dtype=np.float32)  # FAISS는 float32 필요
    
    logger.info("벡터 shape: %s", vectors.shape)  # (4, 384) 나와야 함
    
    # FAISS 인덱스 생성
    dim = vectors.shape[1]  # ← 벡터 차원수는 몇 번째 축?
    index = faiss.IndexFlatIP(dim)  # ← dim 넣기
    
    # 정규화 (코사인 유사도를 위해 필수)
    faiss.normalize_L2(vectors)
    index.add(vectors)  # ← 벡터를 인덱스에 추가하는 메서드는?
    
    logger.info("FAISS 인덱스 구축 완료: %d개 등록", index.ntotal)
    return index, vectors

def retrieve(query: str, index, docs: list[str], model, top_k: int = 2) -> list[str]:
    """질문과 의미적으로 가까운 문서를 반환한다."""
    top_k = min(top_k, len(docs))  # 방어 코드 (어제 배운 것!)
    
    # 질문도 같은 모델로 벡터화
    query_vec = model.encode([query])  # ← 리스트로 감싸는 이유?
    query_vec = np.array(query_vec, dtype=np.float32)
    faiss.normalize_L2(query_vec)
    
    # FAISS 검색: distances(유사도), indices(인덱스 번호) 반환
    distances, indices = index.search(query_vec, top_k)  # ← top_k 넣기
    
    logger.info("검색 완료 | 최고 유사도: %.4f", distances[0][0])
    
    return [docs[i] for i in indices[0]]  # ← 첫 번째 쿼리 결과만

def generate_answer(query: str, context_docs: list[str]) -> str:
    context = "\n".join(context_docs)
    return f"[검색된 근거]\n{context}\n\n[답변] 관련 런북을 참고하세요."

if __name__ == "__main__":
    model = SentenceTransformer(MODEL_NAME)  # ← MODEL_NAME 상수 사용
    index, vectors = build_faiss_index(RUNBOOKS, model)
    
    query = "Pod가 계속 재시작돼요"
    retrieved = retrieve(query, index, RUNBOOKS, model, top_k=2)
    answer = generate_answer(query, retrieved)
    
    print(answer)