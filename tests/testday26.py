# 내가 작성한 답
# O(n*k) 버전 — 직관적이지만 k가 클수록 느려짐
# def find_anomaly_windows(logs: list[int], k: int) -> list[int]:
#     result = []
#     for index in range(len(logs) - k + 1) :
#         if sum(logs[index:index + k])/k > 0.5:
#             result.append(index)
    
#     return result

# test = [0,0,1,0,1,1,0,0,1,0]
# print(find_anomaly_windows(test,3))

# 계산로직이 더 좋은 답
# O(n) 버전 — 슬라이딩 윈도우: 앞을 빼고 뒤를 더하는 방식
def find_anomaly_windows(logs: list[int], k: int) -> list[int]:
    result = []

    # 첫 윈도우 sum은 한 번만 계산
    window_sum = sum(logs[:k])
    if window_sum / k > 0.5:
        result.append(0)

    # 이후엔 앞을 빼고 뒤를 더하기만 해
    for index in range(1, len(logs) - k + 1):
        window_sum += logs[index + k - 1]  # 뒤에서 들어온 값
        window_sum -= logs[index - 1]       # 앞에서 빠진 값
        if window_sum / k > 0.5:
            result.append(index)

    return result

test = [0,0,1,0,1,1,0,0,1,0]
print(find_anomaly_windows(test, 3))  # [2, 3, 5]
print(find_anomaly_windows(test, 5))  # k=5로 바꿔도 정상 
