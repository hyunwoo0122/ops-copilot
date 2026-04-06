import pytest
import torch
import numpy as np
from day32_pytorch_autoencoder import LogAutoEncoder
from day37_dataloader_train import train_with_dataloader
from day38_metrics import compute_scores, evaluate_metrics


@pytest.fixture
def sample_data():
    """정상/이상 데이터 fixture"""
    torch.manual_seed(42)
    np.random.seed(42)
    normal  = torch.FloatTensor(np.random.randn(200, 8) * 0.5)
    anomaly = torch.FloatTensor(np.random.randn(50,  8) * 3.0)
    return normal, anomaly


@pytest.fixture
def trained_model(sample_data, tmp_path):
    """학습된 모델 fixture"""
    normal, _ = sample_data
    model = LogAutoEncoder(input_dim=8, hidden_dim=4)
    save_path = str(tmp_path / "model.pt")
    train_with_dataloader(
        model, normal,
        epochs=30, patience=5,
        save_path=save_path,
    )
    return model


# 테스트 1: compute_scores shape 확인
def test_compute_scores_shape(sample_data, trained_model):
    """scores shape이 (n_samples,) 이어야 함"""
    normal, _ = sample_data
    scores = compute_scores(trained_model, normal)

    # TODO 1: scores shape 확인
    # 힌트: len(normal) 개수와 같아야 해
    assert scores.shape == (len(normal), )


# 테스트 2: 이상 점수 > 정상 점수
def test_anomaly_scores_higher(sample_data, trained_model):
    """이상 점수 평균이 정상보다 높아야 함"""
    normal, anomaly = sample_data
    normal_scores  = compute_scores(trained_model, normal)
    anomaly_scores = compute_scores(trained_model, anomaly)

    # TODO 2: 평균 비교
    assert np.mean(normal_scores) < np.mean(anomaly_scores)


# 테스트 3: best_threshold가 thresholds 목록 안에 있는지
def test_best_threshold_in_list(sample_data, trained_model):
    """최적 threshold가 반드시 목록 안에 있어야 함"""
    normal, anomaly = sample_data
    thresholds = [0.1, 0.3, 0.5, 0.7, 1.0]

    best_threshold, _ = evaluate_metrics(
        trained_model, normal, anomaly, thresholds
    )

    # TODO 3: best_threshold가 thresholds 안에 있는지 확인
    assert best_threshold in thresholds


# 테스트 4: F1 성능 기준선
def test_f1_above_baseline(sample_data, trained_model):
    """최적 F1이 0.8 이상이어야 함"""
    normal, anomaly = sample_data
    thresholds = [0.1, 0.3, 0.5, 0.7, 1.0]

    _, best_metrics = evaluate_metrics(
        trained_model, normal, anomaly, thresholds
    )

    # TODO 4: F1 기준선 확인
    assert best_metrics['f1'] >= 0.8