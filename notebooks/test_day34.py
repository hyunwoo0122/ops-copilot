import pytest
import torch
import numpy as np
from day32_pytorch_autoencoder import LogAutoEncoder
from day33_logbert_train import train_with_early_stopping, evaluate_model

@pytest.fixture
def sample_data():
    """정상/이상 데이터 fixture"""
    torch.manual_seed(42)
    normal  = torch.FloatTensor(np.random.randn(50, 8) * 0.5)
    anomaly = torch.FloatTensor(np.random.randn(10, 8) * 3.0)
    return normal, anomaly

@pytest.fixture
def trained_model(sample_data):
    """학습된 모델 fixture"""
    normal, _ = sample_data
    model = LogAutoEncoder(input_dim=8, hidden_dim=4)
    train_with_early_stopping(
        model, normal, epochs=20, patience=5
    )
    return model


# 테스트 1: 출력 shape 확인
def test_output_shape():
    """AutoEncoder 입력/출력 shape이 같아야 함"""
    model = LogAutoEncoder(input_dim=8, hidden_dim=4)
    x = torch.randn(10, 8)

    # TODO 1: forward 실행 후 shape 확인
    # 힌트: output.shape == x.shape
    output = model(x)
    assert output.shape == x.shape


# 테스트 2: 학습 후 Loss 감소 확인
def test_loss_decreases(sample_data):
    """학습할수록 Loss가 줄어야 함"""
    normal, _ = sample_data
    model = LogAutoEncoder(input_dim=8, hidden_dim=4)

    losses = train_with_early_stopping(
        model, normal, epochs=20, patience=5
    )

    # TODO 2: 첫 loss > 마지막 loss 확인
    assert losses[0] > losses[-1]


# 테스트 3: 이상 오류 > 정상 오류
def test_anomaly_score_higher(sample_data, trained_model):
    """학습 후 이상 오류가 정상보다 높아야 함"""
    normal, anomaly = sample_data

    # TODO 3: evaluate_model 호출
    # 힌트: normal_err, anomaly_err = evaluate_model(???)
    normal_err, anomaly_err = evaluate_model(trained_model, normal, anomaly)

    assert anomaly_err > normal_err


# 테스트 4: 모델 저장/불러오기
def test_model_save_load(sample_data, tmp_path):
    """저장하고 불러온 모델이 같은 결과를 내야 함"""
    normal, _ = sample_data
    model = LogAutoEncoder(input_dim=8, hidden_dim=4)
    train_with_early_stopping(model, normal, epochs=10, patience=3)

    # 임시 경로에 저장
    save_path = tmp_path / "test_model.pt"
    torch.save(model.state_dict(), save_path)

    # 새 모델에 불러오기
    model2 = LogAutoEncoder(input_dim=8, hidden_dim=4)
    # TODO 4: 저장된 가중치 불러오기
    model2.load_state_dict(torch.load(save_path))

    # 같은 입력에 같은 출력이 나와야 해
    x = torch.randn(5, 8)
    model.eval()
    model2.eval()
    with torch.no_grad():
        # TODO 5: 두 모델 출력이 거의 같은지 확인
        # 힌트: torch.allclose(out1, out2)
        out1 = model(x)
        out2 = model2(x)
        assert torch.allclose(out1, out2)