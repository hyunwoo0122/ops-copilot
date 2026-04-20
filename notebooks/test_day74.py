import pytest
from unittest.mock import patch, MagicMock
import json

# day71_sqs.py에서 가져오기
from day71_sqs import send_message, receive_message, delete_message,QUEUE_URL

def test_send_message():
    # SQS send_message를 Mock으로 대체
    with patch("day71_sqs.sqs") as mock_sqs:
        mock_sqs.send_message.return_value = {
            "MessageId": "test-id-123"
        }
        result = send_message("Podが起動しない")
        assert result == "test-id-123"  # 반환된 MessageId
        mock_sqs.send_message.assert_called_once()

def test_receive_message_empty():
    with patch("day71_sqs.sqs") as mock_sqs:
        # 메시지 없을 때
        mock_sqs.receive_message.return_value = {}
        result = receive_message()
        assert result == []   # 빈 리스트

def test_receive_message_with_data():
    with patch("day71_sqs.sqs") as mock_sqs:
        mock_sqs.receive_message.return_value = {
            "Messages": [
                {
                    "Body": json.dumps({"query": "테스트"}),
                    "ReceiptHandle": "handle-123"
                }
            ]
        }
        result = receive_message()
        assert len(result) == 1   # 메시지 1개
        assert json.loads(result[0]["Body"])["query"] == "테스트"

def test_delete_message():
    with patch("day71_sqs.sqs") as mock_sqs:
        delete_message("handle-123")
        mock_sqs.delete_message.assert_called_once_with(
            QueueUrl= QUEUE_URL,
            ReceiptHandle="handle-123"
        )