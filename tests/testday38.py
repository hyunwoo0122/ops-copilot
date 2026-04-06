

def find_best_threshold(normal: list[float], anomaly: list[float], thresholds: list[float]) -> float:
    best_acc=0
    best_t=thresholds[0]
    count = 0
    for num in thresholds:
        for i in range(len(normal)):
            if num > normal[i]:
                count += 1
        for j in range(len(anomaly)):
            if num < anomaly[j]:
                count += 1
        result = count/(len(normal) + len(anomaly))
        
        if best_acc < result:
            best_acc = result    
            best_t=num 
        count = 0        

    return best_t

normal = [0.1, 0.2, 0.15, 0.3]
anomaly = [0.8, 0.9, 0.7, 0.6]
thresholds = [0.3, 0.5, 0.7]
print(find_best_threshold(normal,anomaly,thresholds))

