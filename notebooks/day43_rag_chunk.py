# day43_rag_chunk.py
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
import faiss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
THRESHOLD = 0.15  # 이 점수 미만이면 관련 문서 없음으로 판단

# 어제보다 긴 런북 (chunking 효과를 보기 위해)
RUNBOOKS = [
    """OOMKilled 발생 시 대응 가이드:
    1단계: kubectl describe pod 명령으로 메모리 사용량을 확인한다.
    2단계: Pod의 memory limit을 현재값의 1.5배로 늘린다.
    3단계: 메모리 프로파일링 도구로 누수 여부를 점검한다.""",

    """CrashLoopBackOff 발생 시 대응 가이드:
    1단계: kubectl logs 명령으로 에러 로그를 확인한다.
    2단계: liveness probe 설정이 올바른지 점검한다.
    3단계: 컨테이너 재시작 간격을 늘려 임시 조치한다.""",

    """DB 응답 지연 발생 시 대응 가이드:
    1단계: slow query log를 확인하고 병목 쿼리를 찾는다.
    2단계: 해당 테이블의 인덱스 적용 여부를 점검한다.
    3단계: 커넥션 풀 설정을 확인하고 필요 시 늘린다.""",
]

def chunk_text(text: str, chunk_size: int = 100, overlap: int = 20) -> list[str]:
    """
    텍스트를 chunk_size 글자씩 자르되,
    overlap만큼 겹치게 해서 경계 잘림을 방지한다.
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size  # chunk_size만큼 자르기
        chunk = text[start:end].strip()  # 앞뒤 공백 제거 (힌트: 문자열 메서드)
        
        if chunk:  # 빈 청크 제외
            chunks.append(chunk)
        
        start += chunk_size - overlap  # overlap만큼 뒤로 돌아가서 다음 시작점
    
    return chunks

def build_faiss_index(docs: list[str], model) -> tuple:
    """문서를 청킹 후 임베딩해서 FAISS 인덱스를 구축한다."""
    all_chunks = []
    
    for doc in docs:
        chunks = chunk_text(doc)  # 문서를 청크로 분할
        all_chunks.extend(chunks)
        logger.info("청크 생성: %d개", len(chunks))
    
    logger.info("전체 청크 수: %d개", len(all_chunks))
    
    vectors = model.encode(all_chunks)
    vectors = np.array(vectors, dtype=np.float32)
    
    dim = vectors.shape[1]  # 어제 배운 것!
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(vectors)
    index.add(vectors)
    
    return index, all_chunks  # docs 대신 all_chunks 반환

def retrieve(
    query: str,
    index,
    chunks: list[str],
    model,
    top_k: int = 2,
    threshold: float = THRESHOLD  # ← threshold 추가
) -> list[str]:
    """유사도 threshold 이상인 청크만 반환한다."""
    top_k = min(top_k, len(chunks))
    
    query_vec = model.encode([query])
    query_vec = np.array(query_vec, dtype=np.float32)
    faiss.normalize_L2(query_vec)
    
    distances, indices = index.search(query_vec, top_k)
    
    results = []
    for score, idx in zip(distances[0], indices[0]):
        if score >= threshold:  # threshold 미만이면 제외
            results.append(chunks[idx])
            logger.info("채택: 유사도=%.4f | %s...", score, chunks[idx][:30])
        else:
            logger.info("제외: 유사도=%.4f (threshold=%.2f 미만)", score, threshold)
    
    if not results:
        return ["관련 런북을 찾을 수 없습니다. 운영팀에 문의하세요."]
    
    return results  # results 반환

def generate_answer(query: str, context_docs: list[str]) -> str:
    context = "\n".join(context_docs)
    return f"[검색된 근거]\n{context}\n\n[답변] 관련 런북을 참고하세요."

if __name__ == "__main__":
    model = SentenceTransformer(MODEL_NAME)
    index, chunks = build_faiss_index(RUNBOOKS, model)
    
    # 테스트 1: 관련 있는 쿼리
    query1 = "Pod가 계속 재시작돼요"
    retrieved1 = retrieve(query1, index, chunks, model, top_k=2)
    print("\n=== 쿼리 1 ===")
    print(generate_answer(query1, retrieved1))
    
    # 테스트 2: 관련 없는 쿼리 (threshold 필터링 확인)
    query2 = "오늘 점심 뭐 먹을까요"
    retrieved2 = retrieve(query2, index, chunks, model, top_k=2)
    print("\n=== 쿼리 2 ===")
    print(generate_answer(query2, retrieved2))