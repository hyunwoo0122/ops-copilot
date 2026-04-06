# def find_K(scores : list[float], k: int) -> list[int]:
#     dic = {}
#     i = 0
#     answer = []
#     for num in scores:
#         dic[num] = i
#         i += 1
#     dic = {k: dic[k] for k in sorted(dic,reverse=True)}
#     items = list(dic.items())[:k]

#     for key, value in items:
#         answer.append(value)
    
#     return answer

# 네 코드 — 12줄, 딕셔너리, 충돌 위험
def find_K(scores, k):
    dic = {}
    i = 0
    for num in scores:
        dic[num] = i
        i += 1
    dic = {k: dic[k] for k in sorted(dic, reverse=True)}
    items = list(dic.items())[:k]
    answer = []
    for key, value in items:
        answer.append(value)
    return answer

# 권장 코드 — 3줄, 인덱스 기준 정렬, 충돌 없음
def find_K(scores, k):
    indices = list(range(len(scores)))
    sorted_indices = sorted(indices, key=lambda i: scores[i], reverse=True)
    return sorted_indices[:k]

scores = [0.3, 0.7, 0.55, 0.9, 0.2]
k = 3
print(find_K(scores,k))