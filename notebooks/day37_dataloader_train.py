import torch
import torch.nn as nn
import numpy as np
import mlflow
import logging
from torch.utils.data import DataLoader, TensorDataset
from day32_pytorch_autoencoder import LogAutoEncoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def train_with_dataloader(
    model: nn.Module,
    data: torch.Tensor,
    epochs: int = 50,
    lr: float = 0.01,
    batch_size: int = 32,
    patience: int = 5,
    save_path: str = "best_model_v2.pt",
) -> list[float]:
    """
    DataLoader 기반 미니배치 학습 함수

    Args:
        model     : 학습할 모델
        data      : 정상 로그 텐서
        epochs    : 최대 학습 횟수
        lr        : 학습률
        batch_size: 배치 크기
        patience  : Early Stopping patience
        save_path : 모델 저장 경로

    Returns:
        epoch_losses: 에폭별 평균 loss 리스트
    """
    # TODO 1: TensorDataset + DataLoader 생성
    # 힌트: TensorDataset(data) → DataLoader(..., batch_size=?, shuffle=?)
    dataset = TensorDataset(data)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    patience_counter = 0
    epoch_losses = []

    for epoch in range(epochs):
        model.train()
        batch_losses = []

        # TODO 2: 배치 학습 루프
        for batch in loader:
            # TODO 3: 배치에서 데이터 꺼내기
            # 힌트: x = batch[0]
            x = batch[0]

            # 학습 루프 5단계 (Day 32에서 배운 것)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, x)
            loss.backward()
            optimizer.step()

            # TODO 4: 배치 loss 저장
            batch_losses.append(loss.item())

        # TODO 5: epoch 평균 loss 계산
        # 힌트: np.mean(batch_losses)
        epoch_loss = np.mean(batch_losses)
        epoch_losses.append(epoch_loss)

        # Early Stopping (Day 33에서 배운 것)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            logging.info(
                f"Epoch {epoch+1:2d} | Loss: {epoch_loss:.4f} ✅ 저장"
            )
        else:
            patience_counter += 1
            logging.info(
                f"Epoch {epoch+1:2d} | Loss: {epoch_loss:.4f} "
                f"(개선없음 {patience_counter}/{patience})"
            )
            if patience_counter >= patience:
                logging.info(f"Early Stopping! {epoch+1}번째 에폭에서 종료")
                break

    return epoch_losses


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # 데이터 크기를 키워봐 — DataLoader의 진가 발휘!
    normal_data  = torch.FloatTensor(
        np.random.randn(1000, 8) * 0.5   # 7주차 100개 → 1000개로!
    )
    anomaly_data = torch.FloatTensor(
        np.random.randn(100, 8) * 3.0
    )

    model = LogAutoEncoder(input_dim=8, hidden_dim=4)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("ops-copilot")

    with mlflow.start_run(run_name="dataloader_batch32"):
        mlflow.log_params({
            "batch_size": 32,
            "epochs"    : 50,
            "lr"        : 0.01,
            "data_size" : 1000,
            "patience"  : 5,
        })

        losses = train_with_dataloader(
            model, normal_data,
            epochs=50, batch_size=32, patience=5
        )

        for i, loss in enumerate(losses):
            mlflow.log_metric("epoch_loss", loss, step=i)

        # 결과 확인
        model.load_state_dict(torch.load("best_model_v2.pt"))
        model.eval()
        with torch.no_grad():
            normal_err  = torch.mean(
                (normal_data - model(normal_data)) ** 2
            ).item()
            anomaly_err = torch.mean(
                (anomaly_data - model(anomaly_data)) ** 2
            ).item()

        mlflow.log_metric("normal_error",  normal_err)
        mlflow.log_metric("anomaly_error", anomaly_err)

        logging.info(f"정상 오류 : {normal_err:.4f}")
        logging.info(f"이상 오류 : {anomaly_err:.4f}")
        logging.info(f"배율      : {anomaly_err/normal_err:.1f}배")