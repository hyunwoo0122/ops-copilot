# 함수로 만들기
# 리스트에 넣어서 확인
# 먼저 괄호를 넣고 만약 이

def count_log_levels(logs: list) -> tuple[dict[str, int], str]:
    counts = {}
    for level in logs:
        # if level in counts:
        #     counts[level] += 1
        # else :
        #     counts[level] = 1
        counts[level] = counts.get(level, 0) + 1

    return counts, max(counts, key=counts.get)
    
test = ["INFO", "ERROR", "INFO", "WARNING", "ERROR", "ERROR"]
print(count_log_levels(test))
