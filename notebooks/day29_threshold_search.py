import numpy as np
import mlflow
from sklearn.metrics import precision_recall_fscore_support
from day28_logbert_proto import compute_anomaly_score

np.random.seed(42)
normal  = [np.random.randn(5, 8) * 0.5 for _ in range(10)]
anomaly = [np.random.randn(5, 8) * 3.0 for _ in range(10)]

# 정답 레이블: 정상=0, 이상=1
y_true = [0]*10 + [1]*10
all_data = normal + anomaly

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ops-copilot")

best_f1, best_threshold = 0, 0

for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred = [
        int(compute_anomaly_score(x, threshold)[1])
        for x in all_data
    ]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    with mlflow.start_run(run_name=f"threshold_{threshold}"):
        mlflow.log_params({"threshold": threshold})
        mlflow.log_metrics({
            "precision": precision,
            "recall"   : recall,
            "f1"       : f1,
        })

    if f1 > best_f1:
        best_f1, best_threshold = f1, threshold

print(f"최적 threshold: {best_threshold} (F1={best_f1:.4f})")