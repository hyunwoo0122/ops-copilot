import torch
import torch.nn as nn
import mlflow
import logging
from day32_pytorch_autoencoder import LogAutoEncoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def train_with_early_stopping(
    model: nn.Module,
    data: torch.Tensor,
    epochs: int = 50,
    lr: float = 0.01,
    patience: int = 3,
    save_path: str = "best_model.pt",
) -> list[float]:
    """
    Early Stopping + 모델 저장이 포함된 학습 함수

    Args:
        model    : 학습할 모델
        data     : 정상 로그 텐서
        epochs   : 최대 학습 횟수
        lr       : 학습률
        patience : 개선 없을 때 몇 번까지 기다릴지
        save_path: 모델 저장 경로

    Returns:
        losses: 에폭별 loss 리스트
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # TODO 1: Early Stopping 변수 초기화
    # 힌트: best_loss = float('inf'), patience_counter = 0
    best_loss = float('inf')
    patience_counter = 0
    losses = []

    for epoch in range(epochs):
        # 학습 루프 (어제랑 동일)
        model.train()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        # TODO 2: Early Stopping 체크
        if loss.item() < best_loss:
            # TODO 3: best_loss 업데이트 + 모델 저장
            best_loss = loss.item()
            patience_counter = 0
            # 힌트: torch.save(model.state_dict(), save_path)
            torch.save(model.state_dict(), save_path)
            logging.info(f"Epoch {epoch+1:2d} | Loss: {loss.item():.4f} ✅ 저장")
        else:
            # TODO 4: patience_counter 증가 + 종료 체크
            patience_counter += 1
            logging.info(
                f"Epoch {epoch+1:2d} | Loss: {loss.item():.4f} "
                f"(개선없음 {patience_counter}/{patience})"
            )
            if patience_counter >= patience:
                logging.info(f"Early Stopping! {epoch+1}번째 에폭에서 종료")
                break

    return losses


def evaluate_model(
    model: nn.Module,
    normal_data: torch.Tensor,
    anomaly_data: torch.Tensor,
    load_path: str = "best_model.pt",
) -> tuple[float, float]:
    """
    저장된 최적 모델로 정상/이상 재구성 오류를 계산합니다.

    Returns:
        (정상 오류, 이상 오류)
    """
    # TODO 5: 저장된 모델 불러오기
    # 힌트: model.load_state_dict(torch.load(load_path))
    model.load_state_dict(torch.load(load_path))
    model.eval()

    with torch.no_grad():
        normal_score = torch.mean(
            (normal_data - model(normal_data)) ** 2
        ).item()
        anomaly_score = torch.mean(
            (anomaly_data - model(anomaly_data)) ** 2
        ).item()

    return normal_score, anomaly_score


if __name__ == "__main__":
    torch.manual_seed(42)

    normal_data  = torch.FloatTensor(
        __import__('numpy').random.randn(100, 8) * 0.5
    )
    anomaly_data = torch.FloatTensor(
        __import__('numpy').random.randn(10, 8) * 3.0
    )

    model = LogAutoEncoder(input_dim=8, hidden_dim=4)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("ops-copilot")

    with mlflow.start_run(run_name="logbert_early_stopping"):
        mlflow.log_params({
            "epochs"  : 50,
            "lr"      : 0.01,
            "patience": 3,
        })

        losses = train_with_early_stopping(
            model, normal_data, epochs=50, patience=3
        )

        for i, loss in enumerate(losses):
            mlflow.log_metric("train_loss", loss, step=i)

        normal_err, anomaly_err = evaluate_model(
            model, normal_data, anomaly_data
        )

        mlflow.log_metric("normal_error",  normal_err)
        mlflow.log_metric("anomaly_error", anomaly_err)

        logging.info(f"정상 오류: {normal_err:.4f}")
        logging.info(f"이상 오류: {anomaly_err:.4f}")
        logging.info(f"총 학습 에폭: {len(losses)}")