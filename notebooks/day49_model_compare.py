# day49_model_compare.py
import logging
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
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── 비교할 모델 2개 ───
MODELS = {
    "다국어": "paraphrase-multilingual-MiniLM-L12-v2",
    "한국어": "snunlp/KR-SBERT-V40K-klueNLI-augSTS",  # 한국어 특화
}

def compare_models(models: dict, runbooks: list, eval_set: list) -> dict:
    """
    여러 모델의 정확도를 비교한다.
    """
    results = {}

    for model_name, model_path in models.items():  # 딕셔너리 순회
        logger.info("모델 로드: %s", model_path)

        try:
            model = SentenceTransformer(model_path)  # model_path 사용
            index, chunks = build_faiss_index(runbooks, model)
            result = evaluate(eval_set, index, chunks, model)

            results[model_name] = {
                "정확도": result["정확도"],
                "근거율": result["근거율"],
                "모델": model_path,
            }
            logger.info("%s 정확도: %.2f%%",
                       model_name, result["정확도"] * 100)

        except Exception as e:
            logger.error("모델 로드 실패: %s | 오류: %s", model_path, e)
            results[model_name] = {"정확도": None, "오류": str(e)}

    return results

if __name__ == "__main__":
    results = compare_models(MODELS, RUNBOOKS, EVAL_SET)

    print("\n=== 모델 성능 비교 ===")
    for name, result in results.items():
        if result["정확도"] is not None:
            print(f"{name}: 정확도={result['정확도']:.2%}")
        else:
            print(f"{name}: 로드 실패 ({result['오류']})")