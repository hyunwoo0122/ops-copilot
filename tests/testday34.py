def find_cost(scores):
    dp = [1] * len(scores)

    for i in range(1, len(scores)):
        for j in range(0, i):
            if  scores[j] < scores[i]:
                dp[i] = max(dp[i], dp[j]+1)

    return max(dp)

scores = [0.3, 0.1, 0.5, 0.2, 0.8, 0.4, 0.9]
print(find_cost(scores))

