import pytest
import numpy as np
from day23_mlflow_compare import run_experiment

# ----테스트용 가짜 데이터 (fixture) -------
@pytest.fixture
def sample_data():
    """80개 정상 + 20개 이상 데이터를 반환하는 fixture"""
    np.random.seed(42)
    X = np.vstack([
        np.random.randn(80,3),
        np.random.randn(20,3) + 4,
    ])
    y_true = np.array([0]*80 + [1]*20)
    return X, y_true


# ----테스트 1: 정상 케이스 -------
def test_f1_score_range(sample_data):
    """F1 Scorerk 0~1 사이인지 확인"""
    X, y_true = sample_data

    # TODO 1: run_experiment()를 호출해봐 (contamination=0.2)
    f1 = run_experiment(X, y_true, contamination=0.2)
                        
    # TODO 2: assert로 범위 확인
    # 힌트: "f1이 0보다 크다"와 "f1이 1 이하다"를 각각 assert로 써봐
    assert f1 > 0
    assert f1 <= 1

# ----테스트 2: 경계값 -------   
def test_min_contamination(sample_data):
    """contamination 최솟값(0.01)에서도 동작하는지 확인"""
    X, y_true = sample_data

    # TODO 3: contamination=0.01 로 실행하고 f1 >= 0 확인
    f1 = run_experiment(X, y_true, 0.01)
    assert f1 >= 0

# ----테스트 3: 엣지케이스 -------   
def test_all_normal_data():
    """이상 데이터가 없을 떄 에러 없이 동작하는지 확인"""
    np.random.seed(0)
    X_all_normal = np.random.randn(50,3)
    y_all_normal = np.array([0]*50)

    # TODO 4: 에러 없이 float를 반환하는지 확인
    # 힌트: max(0.01, contamination) 버그 기억해? (Day 1개월차)
    f1 = run_experiment(X_all_normal, y_all_normal, contamination=0.01)
    assert isinstance(f1, float)