import boto3
import json
import time

sqs = boto3.client("sqs", region_name="ap-northeast-1")
QUEUE_URL = "https://sqs.ap-northeast-1.amazonaws.com/106281192812/ops-copilot-queue"

def process_message(body: dict):
    query = body.get("query")   # query 추출
    print(f"처리 중: {query}")
    time.sleep(0.1)              # Mock 처리
    print(f"처리 완료: {query}")

def run_worker():
    print("Worker 시작 — SQS 대기 중...")
    while True:                 # 무한 루프
        messages = sqs.receive_message(
            QueueUrl=QUEUE_URL,
            MaxNumberOfMessages=10,
            WaitTimeSeconds=5
        ).get("Messages", [])

        if not messages:
            print("메시지 없음 — 대기 중...")
            continue

        for msg in messages:
            body = json.loads(msg["Body"])   # JSON 파싱
            process_message(body)
            sqs.delete_message(
                QueueUrl=QUEUE_URL,
                ReceiptHandle=msg["ReceiptHandle"]
            )
            print(f"삭제 완료: {msg['MessageId']}")

if __name__ == "__main__":
    run_worker()