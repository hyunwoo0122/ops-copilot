# def sum_scores(scores: list[float], target: float) -> int | int | None:
#     length = len(scores)
#     left = 0
#     right = length - 1
#     result = 0

#     for num in range(length):
#         result = scores[left] + scores[right]
#         if result > target:
#             right = right - 1
#             result = scores[left] + scores[right]
#             print(scores[left], scores[right])
#         elif result < target:
#             left = left + 1
#             result = scores[left] + scores[right]
#             print(scores[left], scores[right])
#         else:
#             print(scores[left], scores[right])
#             return left,right

#     return None

def sum_scores(scores: list[float], target: float) -> tuple[int, int] | None:
    left = 0
    right = len(scores) - 1

    while left < right:          # for문 말고 while!
        current = scores[left] + scores[right]
        if current == target:
            return left, right   # 바로 반환
        elif current > target:
            right -= 1           # 오른쪽 줄이기
        else:
            left += 1            # 왼쪽 키우기

    return None

scores = [0.1, 0.3, 0.5, 0.7, 0.9]
queries = 1.0

print(sum_scores(scores, queries))


