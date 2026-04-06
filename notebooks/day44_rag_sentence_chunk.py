# day44_rag_sentence_chunk.py
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
import faiss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
THRESHOLD = 0.15

RUNBOOKS = [
    """OOMKilled 발생 시 대응 가이드:
    1단계: kubectl describe pod 명령으로 메모리 사용량을 확인한다.
    2단계: Pod의 memory limit을 현재값의 1.5배로 늘린다.
    3단계: 메모리 프로파일링 도구로 누수 여부를 점검한다.""",

    """CrashLoopBackOff 발생 시 대응 가이드:
    1단계: kubectl logs 명령으로 에러 로그를 확인한다.
    2단계: liveness probe 설정이 올바른지 점검한다.
    3단계: 컨테이너 재시작 간격을 늘려 임시 조치한다.""",
]

def chunk_by_sentence(text: str) -> list[str]:
    """
    \n 기준으로 문장을 나누고 빈 줄을 제거한다.
    어제의 고정 글자수 방식에서 개선된 버전.
    """
    lines = text.split('\n') # \n으로 분리하는 메서드
    chunks = []
    for line in lines:
        stripped = line.strip()    # 앞뒤 공백 제거
        if stripped:           # 빈 줄 제외
            chunks.append(stripped)
    return chunks

def build_faiss_index(docs: list[str], model) -> tuple:
    all_chunks = []
    for doc in docs:
        chunks = chunk_by_sentence(doc)
        all_chunks.extend(chunks)
        logger.info("문장 청크 생성: %d개", len(chunks))

    logger.info("전체 청크 수: %d개", len(all_chunks))

    vectors = model.encode(all_chunks)
    vectors = np.array(vectors, dtype=np.float32)
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index, all_chunks

def retrieve(query, index, chunks, model, top_k=2, threshold=THRESHOLD):
    top_k = min(top_k, len(chunks))
    query_vec = np.array(model.encode([query]), dtype=np.float32)
    faiss.normalize_L2(query_vec)

    distances, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(distances[0], indices[0]):
        if score >= threshold:
            results.append(chunks[idx])
            logger.info("채택: %.4f | %s...", score, chunks[idx][:30])
        else:
            logger.info("제외: %.4f", score)

    return results if results else ["관련 런북을 찾을 수 없습니다."]

if __name__ == "__main__":
    model = SentenceTransformer(MODEL_NAME)
    index, chunks = build_faiss_index(RUNBOOKS, model)

    query = "Pod가 계속 재시작돼요"
    result = retrieve(query, index, chunks, model)
    print("\n".join(result))