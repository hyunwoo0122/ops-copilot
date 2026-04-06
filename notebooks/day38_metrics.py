import torch
import numpy as np
import mlflow
import logging
from sklearn.metrics import precision_recall_fscore_support
from day32_pytorch_autoencoder import LogAutoEncoder
from day37_dataloader_train import train_with_dataloader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def compute_scores(
    model: torch.nn.Module,
    data: torch.Tensor,
) -> np.ndarray:
    """
    각 데이터의 재구성 오류(이상 점수)를 계산합니다.

    Returns:
        scores: 각 샘플의 재구성 오류 배열
    """
    model.eval()
    with torch.no_grad():
        output = model(data)
        # TODO 1: 샘플별 MSE 계산
        # 힌트: 각 행(axis=1)의 평균 → shape (n_samples,)
        # torch.mean((data - output) ** 2, dim=???)
        scores = torch.mean((data - output) ** 2, dim=1).numpy()
    return scores

def evaluate_metrics(
    model: torch.nn.Module,
    normal_data: torch.Tensor,
    anomaly_data: torch.Tensor,
    thresholds: list[float],
) -> tuple[float, dict]:
    """
    threshold별 Recall, FPR, F1을 계산하고 최적 threshold를 반환합니다.

    Returns:
        best_threshold: F1이 최대인 threshold
        best_metrics  : 최적 threshold의 지표 딕셔너리
    """
    # 이상 점수 계산
    normal_scores  = compute_scores(model, normal_data)
    anomaly_scores = compute_scores(model, anomaly_data)

    # TODO 2: 정답 레이블 만들기
    # 정상=0, 이상=1
    y_true = np.array(
        [0] * len(normal_scores) + [1] * len(anomaly_scores)
    )
    all_scores = np.concatenate([normal_scores, anomaly_scores])

    best_f1 = 0
    best_threshold = thresholds[0]
    best_metrics = {}

    for threshold in thresholds:
        # TODO 3: threshold로 이상/정상 예측
        # 힌트: 이상 점수 > threshold 이면 1, 아니면 0
        y_pred = (all_scores > threshold).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred,
            average="binary",
            zero_division=0,
        )

        # TODO 4: FPR 직접 계산
        # FPR = FP / (FP + TN)
        # 힌트: 정상(y_true=0) 중에서 이상으로 예측(y_pred=1)된 것
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        logging.info(
            f"threshold={threshold:.2f} | "
            f"Recall={recall:.3f} FPR={fpr:.3f} F1={f1:.3f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                "precision": precision,
                "recall"   : recall,
                "fpr"      : fpr,
                "f1"       : f1,
            }

    return best_threshold, best_metrics


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    normal_data  = torch.FloatTensor(
        np.random.randn(1000, 8) * 0.5
    )
    anomaly_data = torch.FloatTensor(
        np.random.randn(100, 8) * 3.0
    )

    model = LogAutoEncoder(input_dim=8, hidden_dim=4)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("ops-copilot")

    with mlflow.start_run(run_name="fpr_recall_tuning"):
        # 학습
        train_with_dataloader(
            model, normal_data, epochs=30, patience=5
        )

        # threshold 목록
        thresholds = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

        best_threshold, best_metrics = evaluate_metrics(
            model, normal_data, anomaly_data, thresholds
        )

        # MLflow에 최적 결과 기록
        mlflow.log_param("best_threshold", best_threshold)
        mlflow.log_metrics(best_metrics)

        logging.info(f"최적 threshold : {best_threshold}")
        logging.info(f"최적 F1        : {best_metrics['f1']:.3f}")
        logging.info(f"최적 Recall    : {best_metrics['recall']:.3f}")
        logging.info(f"최적 FPR       : {best_metrics['fpr']:.3f}")