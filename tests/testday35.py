def count_unique_per_batch(data: list[int], k: int) -> list[int]:
    result = []

    for num in range(0,len(data),k):
        result.append(len(set(data[num:num+k])))
    
    return result
    

data = [1,1,2,3,3,3,4,2,2,4]
print(count_unique_per_batch(data,4))

