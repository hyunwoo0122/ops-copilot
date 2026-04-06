def count_above_threshold(scores: list[float], threshold: float) -> int:
    """
    정렬된 배열에서 threshold 이상인 원소 개수를 반환합니다.

    Args:
        scores   : 오름차순 정렬된 이상 점수 배열
        threshold: 기준값

    Returns:
        threshold 이상인 원소 개수
    """
    # O(n) 버전 — 직관적
    # count = sum(1 for s in scores if s >= threshold)

    # O(log n) 버전 — 이진탐색
    left, right = 0, len(scores) - 1

    while left <= right:
        mid = (left + right) // 2
        if scores[mid] < threshold:
            left = mid + 1   # 왼쪽 절반 버리기
        else:
            right = mid - 1  # 오른쪽으로 좁히기

    return len(scores) - left  # left = 처음으로 threshold 이상인 인덱스


# 테스트
scores = [0.1, 0.3, 0.5, 0.6, 0.8, 0.9]
print(count_above_threshold(scores, 0.5))  # 4
print(count_above_threshold(scores, 0.0))  # 6 (전부)
print(count_above_threshold(scores, 1.0))  # 0 (없음)