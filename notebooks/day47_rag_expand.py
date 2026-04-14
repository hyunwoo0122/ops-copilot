# day47_rag_expand.py
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
THRESHOLD = 0.3

# ─── 런북 10개로 확장 ───
RUNBOOKS = [
    """OOMKilled 대응:
    1단계: kubectl describe pod로 메모리 사용량 확인.
    2단계: memory limit을 1.5배로 늘린다.
    3단계: 메모리 누수 여부를 프로파일링한다.""",

    """CrashLoopBackOff 대응:
    1단계: kubectl logs로 에러 로그 확인.
    2단계: liveness probe 설정 점검.
    3단계: 컨테이너 재시작 간격을 늘린다.""",

    """DB 응답 지연 대응:
    1단계: slow query log에서 병목 쿼리 확인.
    2단계: 인덱스 적용 여부 점검.
    3단계: 커넥션 풀 설정 확인.""",

    """디스크 풀 대응:
    1단계: df -h 명령으로 디스크 사용량 확인.
    2단계: 오래된 로그 파일 정리.
    3단계: PVC 용량 확장 요청.""",

    """Pod Pending 대응:
    1단계: kubectl describe pod로 이벤트 확인.
    2단계: 노드 리소스(CPU/메모리) 여유 확인.
    3단계: 노드 추가 또는 리소스 요청량 조정.""",

    """네트워크 타임아웃 대응:
    1단계: 서비스 엔드포인트 연결 상태 확인.
    2단계: NetworkPolicy 설정 점검.
    3단계: DNS 해석 오류 여부 확인.""",

    """CPU 과부하 대응:
    1단계: top 명령으로 CPU 사용률 높은 프로세스 확인.
    2단계: HPA(HorizontalPodAutoscaler) 설정 점검.
    3단계: CPU limit 상향 조정.""",

    """인증 오류(401) 대응:
    1단계: API 토큰 만료 여부 확인.
    2단계: RBAC 권한 설정 점검.
    3단계: 시크릿 재발급 및 재배포.""",

    """배포 실패 대응:
    1단계: kubectl rollout status로 배포 상태 확인.
    2단계: 이미지 태그 및 레지스트리 접근 권한 확인.
    3단계: kubectl rollout undo로 이전 버전으로 롤백.""",

    """서비스 응답 없음(503) 대응:
    1단계: Pod 상태 및 Readiness Probe 확인.
    2단계: 서비스 셀렉터와 Pod 라벨 일치 여부 확인.
    3단계: 로드밸런서 헬스체크 설정 점검.""",
]

# ─── 평가셋 8개로 확장 ───
EVAL_SET = [
    ("Pod가 계속 재시작돼요",    "CrashLoopBackOff 대응"),  # 제목 청크에 있는 단어
    ("메모리가 부족해요",        "메모리 누수"),             # 내용 청크에 있는 단어
    ("DB가 느려요",             "DB 응답 지연 대응"),       # ✅ 이미 맞음
    ("디스크가 꽉 찼어요",       "디스크 풀 대응"),          # ✅ 이미 맞음
    ("Pod가 Pending 상태예요",   "노드 리소스"),             # 내용 청크에 있는 단어
    ("CPU 사용률이 너무 높아요",  "CPU 과부하 대응"),         # ✅ 이미 맞음
    ("배포가 실패했어요",        "배포 실패 대응"),           # ✅ 이미 맞음
    ("오늘 점심 뭐 먹을까요",    None),
]

def chunk_by_sentence(text: str) -> list[str]:
    return [line.strip() for line in text.split('\n') if line.strip()]

def build_faiss_index(docs: list[str], model) -> tuple:
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_by_sentence(doc))
    vectors = np.array(model.encode(all_chunks), dtype=np.float32)
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
    return results

def evaluate(eval_set: list, index, chunks, model) -> dict:
    correct = 0
    faithful = 0
    total = len(eval_set)

    for query, expected in eval_set:
        retrieved = retrieve(query, index, chunks, model)
        context = " ".join(retrieved)

        if retrieved:
            faithful += 1

        if expected is None:
            if not retrieved:
                correct += 1
        else:
            if expected in context:
                correct += 1

        logger.info("쿼리: %s | 정답: %s | 결과: %s",
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
            print(f"{key}: {value:.2%}")
        else:
            print(f"{key}: {value}")