import numpy as np
import mlflow
import logging
from day27_self_attention import scaled_dot_product_attention

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def compute_anomaly_score(
    X: np.ndarray,
    threshold: float = 0.5,
) -> tuple[float, bool]:
    """
    Attention 기반 재구성 오류로 이상 점수를 계산합니다.

    Args:
        X        : 로그 시퀀스 (seq_len, d_model)
        threshold: 이상 판단 임계값

    Returns:
        score   : 이상 점수 (재구성 오류 평균)
        is_anomaly: 임계값 초과 여부
    """
    # TODO 1: Q, K, V 를 X 자체로 설정 (Self-Attention)
    # 힌트: LogBERT에서 Q=K=V=X 야
    Q = K = V = X

    # TODO 2: 어제 만든 함수로 Attention 실행
    output, _ = scaled_dot_product_attention(Q, K, V)

    # TODO 3: 재구성 오류 계산
    # 입력(X)과 출력(output)의 차이의 절댓값 평균
    # 힌트: np.abs(X - output).mean()
    score = np.abs(X - output).mean()

    # TODO 4: 임계값과 비교해서 이상 여부 반환
    is_anomaly = score > threshold

    return score, is_anomaly


if __name__ == "__main__":
    np.random.seed(42)
    d_model = 8

    # 정상 로그: 작은 값들로 구성
    normal_logs = [np.random.randn(5, d_model) * 0.5 for _ in range(5)]
    # 이상 로그: 큰 값들로 구성 (분포가 다름)
    anomaly_logs = [np.random.randn(5, d_model) * 3.0 for _ in range(3)]

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("ops-copilot")

    with mlflow.start_run(run_name="logbert_prototype"):

        normal_scores  = [compute_anomaly_score(x)[0] for x in normal_logs]
        anomaly_scores = [compute_anomaly_score(x)[0] for x in anomaly_logs]

        # TODO 5: MLflow에 평균 점수 기록
        mlflow.log_metric("avg_normal_score",  float(np.mean(normal_scores)))
        mlflow.log_metric("avg_anomaly_score", float(np.mean(anomaly_scores)))

        logging.info(f"정상 평균 점수 : {np.mean(normal_scores):.4f}")
        logging.info(f"이상 평균 점수 : {np.mean(anomaly_scores):.4f}")
        logging.info("이상 점수가 정상보다 높으면 프로토타입 성공!")