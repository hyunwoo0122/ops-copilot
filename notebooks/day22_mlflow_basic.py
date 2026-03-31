import mlflow
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
import numpy as np

# 1) 실험 이름 설정 (mlflow에 "ops-copilot" 이라는 폴더가 생겨)
# TODO: mlflow.set_experiment("???") 에 뭘 넣으면 좋을까?
# 데이터를 어디에, 어떤 폴더에 저장할지 정하는 단계
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ops-copilot")

# 가짜 로그 데이터 (0=정상, 1=이상)
np.random.seed(42)
X_normal   = np.random.randn(80, 3)
X_anomaly  = np.random.randn(20, 3) + 4
X = np.vstack([X_normal, X_anomaly])
y_true = np.array([0]*80 + [1]*20)

# 2) MLflow 실험 시작
# TODO: with mlflow.???(run_name="baseline_IsolationForest"): 뭐가 들어갈까?
# run_name으로 이름을 정해줌
with mlflow.start_run(run_name="baseline_IsolationForest"):

    # 파라미터 설정
    contamination = 0.2
    n_estimators  = 100

    # 3) 파라미터를 mlflow에 기록
    # TODO: mlflow.log_???({"contamination": contamination,
    #                        "n_estimators": n_estimators})
    # 파라미터를 기록하기 위해 작성함
    # 실험 전에 우리가 정한 설정값(Recipe). 나중에 "어떤 설정이 좋았지?" 하고 비교할 때 씁니다.
    mlflow.log_params({"contamination": contamination,
                    "n_estimators": n_estimators})

    # 모델 학습
    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=42
    )
    model.fit(X)
    # IsolationForest기법이 0,-1로 구분해서 -1을 전부 1로 치환해야 하기 때문에 사용
    # Isolation Forest는 "데이터를 분리(Isolate)하기 쉬울수록 이상치다"라는 원리를 가집니다. 그래서 평균에서 멀리 떨어진 녀석들을 -1로 뱉어냅니다.
    preds = (model.predict(X) == -1).astype(int)

    # 4) 결과(점수)를 mlflow에 기록
    f1 = f1_score(y_true, preds)
    # TODO: mlflow.log_???({"f1_score": f1})
    # f1점수를 기록하기 위해서
    # 실험 후에 나온 성적표. 시간이 지나면서 변할 수 있는 수치(학습 곡선 등)를 기록할 때 씁니다.
    mlflow.log_metric("f1_score", f1)

    print(f"F1 Score: {f1:.4f}")
    print("MLflow 기록 완료! 브라우저에서 확인해봐")