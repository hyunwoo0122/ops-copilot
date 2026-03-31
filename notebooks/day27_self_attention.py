import numpy as np
import mlflow
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Softmax 함수 — 점수를 확률(합=1)로 변환

    Args:
        x: 입력 배열

    Returns:
        확률 배열 (각 행의 합 = 1.0)
    """
    # TODO 1: softmax 구현
    # 힌트: exp(x) / sum(exp(x))
    # 수치 안정성을 위해 x - max(x) 를 먼저 빼줘
    # 검색: "softmax numerical stability"
    e_x = np.exp(x - x.max(axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scaled Dot-Product Attention 구현

    Args:
        Q: Query 행렬 (seq_len, d_k)
        K: Key 행렬   (seq_len, d_k)
        V: Value 행렬 (seq_len, d_v)

    Returns:
        output        : Attention 결과 (seq_len, d_v)
        attention_weights: 각 토큰의 집중도 (seq_len, seq_len)
    """
    d_k = Q.shape[-1]

    # TODO 2: Q와 K를 행렬 곱해서 점수 계산
    # 힌트: K를 전치(transpose)해서 곱해야 해
    # 검색: "numpy matmul transpose"
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)

    # TODO 3: softmax 적용 (axis=-1: 각 행마다 적용)
    attention_weights = softmax(scores)

    # TODO 4: attention_weights와 V를 곱해서 최종 출력
    output = np.matmul(attention_weights, V)

    return output, attention_weights


if __name__ == "__main__":
    np.random.seed(42)

    # 가짜 로그 시퀀스: 5개 토큰, 각 토큰은 8차원
    seq_len = 5
    d_k = 8

    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_k)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("ops-copilot")

    with mlflow.start_run(run_name="self_attention_prototype"):
        output, weights = scaled_dot_product_attention(Q, K, V)

        # TODO 5: MLflow에 shape 정보 기록
        # 힌트: seq_len, d_k 를 params로, weights 최댓값을 metric으로
        mlflow.log_params({
            "seq_len": seq_len,
            "d_k"    : d_k,
        })
        mlflow.log_metric("max_attention_weight", float(weights.max()))

        logging.info(f"Output shape : {output.shape}")
        logging.info(f"Weights shape: {weights.shape}")
        logging.info(f"Attention weights (첫 토큰이 어디 집중?):\n{weights[0].round(3)}")