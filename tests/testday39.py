def find_best_count(scores: list[float], k: float) -> int:
    scores = sorted(scores)
    count = 1
    start = scores[0] + k
    for num in scores:
        if num > start:
            count += 1
            start = num + k

    return count

scores = [0.1, 0.4, 0.35, 0.8, 0.75, 0.9]
print(find_best_count(scores,0.3))

