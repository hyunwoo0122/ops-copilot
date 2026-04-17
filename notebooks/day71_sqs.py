import boto3
import json

# SQS 클라이언트 생성
sqs = boto3.client(
    "sqs",              # 서비스 이름
    region_name="ap-northeast-1"   # 도쿄 리전
)

QUEUE_URL = "https://sqs.ap-northeast-1.amazonaws.com/106281192812/ops-copilot-queue"

def send_message(query: str) -> str:
    response = sqs.send_message(
        QueueUrl=QUEUE_URL,
        MessageBody=json.dumps({
            "query": query,
            "timestamp": "2026-04-17"
        })
    )
    return response["MessageId"]   # 메시지 ID 반환

def receive_message() -> list:
    response = sqs.receive_message(
        QueueUrl=QUEUE_URL,
        MaxNumberOfMessages=10,   # 한 번에 최대 10개
        WaitTimeSeconds=5            # 5초 대기 (Long Polling)
    )
    return response.get("Messages", [])

def delete_message(receipt_handle: str):
    sqs.delete_message(
        QueueUrl=QUEUE_URL,
        ReceiptHandle=receipt_handle   # 메시지 고유 핸들
    )

# 테스트
msg_id = send_message("Podが起動しない")
print(f"전송 완료: {msg_id}")

messages = receive_message()
for msg in messages:
    print(f"수신: {msg['Body']}")
    delete_message(msg["ReceiptHandle"])   # 처리 후 삭제
    print("삭제 완료")