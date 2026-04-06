# 함수로 만들기
# 구간을 배열에 넣어
# 그리고 내림 차순으로 정렬하고 2번째 원소 반환

def process_ids(id_list: list[int]) -> int | None:
    """
    정상(0) 구간 길이 목록에서 두 번째로 긴 길이를 반환합니다.

    Args:
        id_list: 0(정상) 또는 1(이상) 배열

    Returns:
        두 번째로 긴 구간 길이. 구간이 2개 미만이면 None.
    """
    total_length = 0
    result = []

    for i, val in enumerate(id_list):      # ← range(len()) 대신 enumerate
        if val == 0:
            total_length += 1
        if i == len(id_list) - 1 or val != 0:
            result.append(total_length)
            total_length = 0

    result = sorted(result, reverse=True)  # ← 루프 밖에서 딱 한 번

    if len(result) < 2:
        return None
    return result[1]


data = [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]
print("정답은:", process_ids(data))  # 2