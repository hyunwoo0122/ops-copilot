import boto3
import time
from datetime import datetime, timezone

# CloudWatch 클라이언트 생성
cw = boto3.client(
    "cloudwatch",              # 서비스 이름
    region_name="ap-northeast-1"   # 도쿄 리전
)

def put_metric(metric_name: str, value: float, unit: str = "Milliseconds"):
    cw.put_metric_data(
        Namespace="OpsCopilot",   # 메트릭 그룹 이름
        MetricData=[
            {
                "MetricName": metric_name,
                "Value": value,
                "Unit": unit,
                "Timestamp": datetime.now(timezone.utc) # 현재시간
            }
        ]
    )

def measure_response_time(func):
    start = time.time()
    result = func()
    elapsed = (time.time() - start) * 1000   # 밀리초 변환
    put_metric("ResponseTime", elapsed)
    return result

# 테스트
def mock_api_call():
    time.sleep(0.1)
    return "ok"

measure_response_time(mock_api_call)
print("메트릭 전송 완료")