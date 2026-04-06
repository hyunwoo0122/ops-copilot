def find_best_count(scores: list[float], thresholds: list[float]) -> dict[float, dict[str, int]]:
    scores = sorted(scores)
    answer = {}
    count = 0
    length = len(scores)
    for num in thresholds:
        result = {}
        count = 0
        for i in range(length):
            if num < scores[i]:
                count += 1
        result["detected"] = count
        result["missed"] = length - count
        answer[num] = result

        
    return answer

scores =  [0.3, 0.7, 0.5, 0.9, 0.2]
thresholds = [0.4, 0.6, 0.8]
print(find_best_count(scores,thresholds))

