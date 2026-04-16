import time
import pytest
from day67_cache import get_cached, set_cache, _cache

# 각 테스트 전에 캐시 초기화
@pytest.fixture(autouse=True)
def clear_cache():
    _cache.clear()   # 딕셔너리 초기화 메서드
    yield

def test_cache_hit_speed():
    # 첫 번째 요청 → 캐시 저장
    set_cache("test_query", "런북 결과")

    # 두 번째 요청 → 캐시 히트
    start = time.time()
    result = get_cached("test_query")
    elapsed = (time.time() - start) * 1000   # 밀리초

    assert result == "런북 결과"
    assert elapsed < 10   # 캐시 히트는 10ms 이내

def test_cache_ttl_expired():
    set_cache("expired_query", "결과")

    # TTL 강제 만료 (저장 시간을 과거로 조작)
    _cache["expired_query"] = ("결과", time.time() - 301)  # 301초 전

    result = get_cached("expired_query")
    assert result is None   # 만료됐으니 None 반환

def test_cache_size():
    set_cache("query1", "결과1")
    set_cache("query2", "결과2")
    set_cache("query3", "결과3")
    assert len(_cache) == 3