# def overlap_str(text: str, size: int, overlap: int) -> str:
#     result = ""
#     answer = []

#     answer.append(text[0:size])

#     for num in range(size,len(text),size):
#         plus = text[num-overlap:num+size-overlap]
#         answer.append(plus)
    
#     return answer

def overlap_str(text: str, size: int, overlap: int) -> list[str]:
    answer = []
    start = size
    
    answer.append(text[0:size])

    while start-overlap < len(text) - (size - overlap) :
        chunk = text[start-overlap:start+size-overlap]
        answer.append(chunk)
        
        start += size - overlap
            
    return answer

text = "ABCDEFGHIJ"
print(overlap_str(text,5,2))
