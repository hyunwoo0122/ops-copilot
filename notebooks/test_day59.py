# test_day59.py
import pytest
from fastapi.testclient import TestClient  # TestClient import
from unittest.mock import patch
from notebooks.day58_fastapi_async import app

# ─── TestClient 생성 ───
client = TestClient(app)  # TestClient(app)

# ─── 테스트 1: 헬스체크 ───
def test_health():
    response = client.get("health") # GET /health
    assert response.status_code == 200  # 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.3.0"
    assert data["chunks"] > 0  # 0보다 커야 함

# ─── 테스트 2: GET 분석 정상 ───
def test_analyze_get_normal():
    with patch("notebooks.day58_fastapi_async.get_intent",
               return_value="CrashLoopBackOff 문제"):
        response = client.get(
            "/analyze",
            params={"query": "Pod가 죽어요"}
        )
    assert response.status_code == 200  # 200
    data = response.json()
    assert "intent" in data     # data에 intent 있는지
    assert "retrieved" in data

# ─── 테스트 3: GET 분석 공백 → 400 ───
def test_analyze_get_empty():
    response = client.get(
        "/analyze",
        params={"query": "   "}  # 공백만
    )
    assert response.status_code == 400  # 400
    assert "query가 비어있습니다" in response.json()["detail"]
    #                                               ↑ "detail"

# ─── 테스트 4: POST 분석 ───
def test_analyze_post():
    with patch("notebooks.day58_fastapi_async.get_intent",
               return_value="OOMKilled 문제"):
        response = client.post(  # POST 메서드
            "/analyze",
            json={"query": "메모리가 부족해요", "top_k": 3}  # 3
        )
    assert response.status_code == 200
    data = response.json()
    assert len(data["retrieved"]) <= 3  # top_k 이하

# ─── 테스트 5: 존재하지 않는 경로 ───
def test_not_found():
    response = client.get("/unknown")
    assert response.status_code == 404  # 404