# def return_batch(scores: list[float], k: int, threshold: float) -> int:
#     answer = 0

#     for num in range(0,len(scores),k):
#         result = (sum(scores[num:num+k]))
#         if result > threshold:
#             answer = num//k
#             return answer
    
#     return -1

def return_batch(scores: list[float], k: int, threshold: float) -> int:
    for i in range(0, len(scores), k):
        if sum(scores[i:i+k]) > threshold:
            return i // k
    return -1

scores =  [0.1, 0.2, 0.8, 0.3, 0.9, 0.7, 0.1, 0.2]
print(return_batch(scores,3,1.5))

