def find_max_subarray(scores: list[float]) -> float:
    max_sum = scores[0]
    current_sum = scores[0]
    index = 1

    while index < len(scores):
        current_sum = max(scores[index], current_sum + scores[index])
        max_sum = max(max_sum, current_sum)
        index += 1

    return max_sum

scores = [-0.2, 0.8, -0.3, 0.9, -0.5, 0.6]
print(find_max_subarray(scores))  # 1.5


