import torch
import torch.nn as nn
import mlflow
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class LogAutoEncoder(nn.Module):
    """
    로그 이상탐지용 AutoEncoder
    정상 로그로 학습 → 이상 로그는 재구성 오류가 커짐
    """
    def __init__(self, input_dim: int = 8, hidden_dim: int = 4):
        # TODO 1: 부모 클래스 초기화 (반드시 필요!)
        # 힌트: super().__???()
        super().__init__()

        # TODO 2: 인코더 정의 (input_dim → hidden_dim)
        self.encoder = nn.Linear(input_dim, hidden_dim)

        # TODO 3: 디코더 정의 (hidden_dim → input_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 로그 텐서 (batch_size, input_dim)
        Returns:
            재구성된 텐서 (batch_size, input_dim)
        """
        # TODO 4: 인코더 → 활성화함수(ReLU) → 디코더 순서로
        encoded = torch.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded


def train_model(
    model: nn.Module,
    data: torch.Tensor,
    epochs: int = 10,
    lr: float = 0.01,
) -> list[float]:
    """
    정상 로그로 AutoEncoder를 학습합니다.

    Args:
        model : 학습할 모델
        data  : 정상 로그 텐서
        epochs: 학습 반복 횟수
        lr    : 학습률

    Returns:
        losses: 에폭별 loss 리스트
    """
    # TODO 5: optimizer와 loss 함수 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []

    for epoch in range(epochs):
        # TODO 6: 학습 루프 3단계
        # 1) optimizer.zero_grad()  → 기울기 초기화
        # 2) output = model(data)   → 순전파
        # 3) loss = criterion(???, ???) → loss 계산
        # 4) loss.backward()        → 역전파
        # 5) optimizer.step()       → 가중치 업데이트
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        logging.info(f"Epoch {epoch+1:2d} | Loss: {loss.item():.4f}")

    return losses


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # 정상 로그 데이터 (100개, 8차원)
    normal_data = torch.FloatTensor(
        np.random.randn(100, 8) * 0.5
    )
    # 이상 로그 데이터 (10개, 8차원)
    anomaly_data = torch.FloatTensor(
        np.random.randn(10, 8) * 3.0
    )

    model = LogAutoEncoder(input_dim=8, hidden_dim=4)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("ops-copilot")

    with mlflow.start_run(run_name="autoencoder_training"):
        mlflow.log_params({
            "input_dim" : 8,
            "hidden_dim": 4,
            "epochs"    : 10,
            "lr"        : 0.01,
        })

        losses = train_model(model, normal_data)

        # MLflow에 각 epoch loss 기록
        for i, loss in enumerate(losses):
            mlflow.log_metric("train_loss", loss, step=i)

        # TODO 7: 학습 후 정상/이상 재구성 오류 계산
        # 힌트: model.eval() 로 추론 모드 전환 먼저!
        model.eval()
        with torch.no_grad():
            normal_recon  = model(normal_data)
            anomaly_recon = model(anomaly_data)

            # TODO 8: MSE 계산
            normal_score  = torch.mean((normal_data  - normal_recon) ** 2)
            anomaly_score = torch.mean((anomaly_data - anomaly_recon) ** 2)

        mlflow.log_metric("normal_recon_error",  normal_score.item())
        mlflow.log_metric("anomaly_recon_error", anomaly_score.item())

        logging.info(f"정상 재구성 오류 : {normal_score.item():.4f}")
        logging.info(f"이상 재구성 오류 : {anomaly_score.item():.4f}")
        logging.info("이상 오류 > 정상 오류 이면 학습 성공!")