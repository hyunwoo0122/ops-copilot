def find_anomaly_windows(logs: list[float], k: int) -> int:
    result = float('inf')
    index = 0

    window_sum = sum(logs[:k])
    if window_sum < result:   
        result = window_sum
        index = 0

    
    for num in range(1, len(logs) - k + 1):
        window_sum += logs[num + k - 1]  
        window_sum -= logs[num - 1]    
        if window_sum < result:
            index = num
            result = window_sum

    return index


data = [0.8, 0.2, 0.1, 0.3, 0.9, 0.4]
print(find_anomaly_windows(data, 3))


