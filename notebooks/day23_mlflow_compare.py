import mlflow
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
import numpy as np
import logging

# TODO 1: logging 설정 — print() 대신 logging.info()를 쓸 거야
# 검색 힌트: "python logging basicConfig level"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ops-copilot")


def run_experiment(X: np.ndarray, y_true: np.ndarray, contamination: float, n_estimators: int = 100) -> float:
    """
    Args:
        X             : 입력 피처 배열 (shape: [n_samples, n_features])
        y_true        : 정답 레이블 배열 (0=정상, 1=이상)
        contamination : 이상 데이터 비율 추정값 (0.0 ~ 0.5)
        n_estimators  : 트리 개수
    """
    # TODO 2: MLflow run 시작 — run_name을 동적으로 만들어봐
    # 예: run_name=f"IF_c{contamination}_n{n_estimators}"
    with mlflow.start_run(run_name=f"IF_c{contamination}_n{n_estimators}"):

        mlflow.log_params({
            "contamination": contamination,
            "n_estimators" : n_estimators,
        })

        model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
        )
        model.fit(X)
        preds = (model.predict(X) == -1).astype(int)

        f1 = f1_score(y_true, preds)
        mlflow.log_metric("f1_score", f1)

        # TODO 3: print() 를 logging.info() 로 바꿔봐
        # 검색 힌트: "python logging info format"
        logging.info(f"실험 완료 - contamination: {contamination}, F1 Score: {f1:.4f}")

        return f1


if __name__ == "__main__":

    np.random.seed(42)
    X = np.vstack([np.random.randn(80, 3), np.random.randn(20, 3) + 4])
    y_true = np.array([0]*80 + [1]*20)

    # TODO 4: 두 가지 contamination 값으로 실험 실행
    # 0.1 과 0.2 를 각각 run_experiment()에 넣어봐
    results = {
        0.1: run_experiment(X, y_true, 0.1),
        0.2: run_experiment(X, y_true, 0.2),
    }
    best = max(results, key=results.get)
    logging.info(f"최적 contamination: {best} (F1={results[best]:.4f})")