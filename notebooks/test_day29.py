import pytest
import numpy as np
from day28_logbert_proto import compute_anomaly_score

@pytest.fixture
def sample_data():
    """정상/이상 데이터 fixture"""
    np.random.seed(42)
    normal  = [np.random.randn(5, 8) * 0.5 for _ in range(5)]
    anomaly = [np.random.randn(5, 8) * 3.0 for _ in range(3)]
    return normal, anomaly


# 테스트 1: 정상 점수 < 이상 점수
def test_anomaly_score_higher(sample_data):
    """이상 점수가 정상보다 높아야 함"""
    normal, anomaly = sample_data

    # TODO 1: 정상/이상 각각 점수 리스트 만들기
    normal_scores  = [compute_anomaly_score(x)[0] for x in normal]
    anomaly_scores = [compute_anomaly_score(x)[0] for x in anomaly]

    # TODO 2: 정상 평균 < 이상 평균 확인
    assert np.mean(normal_scores) < np.mean(anomaly_scores)


# 테스트 2: 점수가 0 이상인지
def test_score_non_negative(sample_data):
    """점수는 항상 0 이상이어야 함 (절댓값 평균이므로)"""
    normal, _ = sample_data
    for x in normal:
        score, _ = compute_anomaly_score(x)
        # TODO 3: score가 0 이상인지 확인
        assert score >= 0


# 테스트 3: threshold 바뀌면 is_anomaly 바뀌는지
def test_threshold_effect():
    """threshold가 낮으면 이상 탐지, 높으면 정상으로 판단"""
    np.random.seed(0)
    X = np.random.randn(5, 8) * 1.0

    # TODO 4: threshold=0.1 이면 is_anomaly=True 여야 해
    _, is_anomaly_low = compute_anomaly_score(X, threshold=0.1)
    assert is_anomaly_low == True

    # TODO 5: threshold=0.9 이면 is_anomaly=False 여야 해
    _, is_anomaly_high = compute_anomaly_score(X, threshold=0.9)
    assert is_anomaly_high == False