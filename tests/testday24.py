# 함수로 만들기
# 리스트에 넣어서 확인
# 먼저 괄호를 넣고 만약 이

def is_valid_bracket(s: str) -> bool:
    stack = []
    if not s:
        return True
    
    for val in s:
        if val == "(" :
            stack.append(val)
        else :
            if stack:
                stack.pop()
            else : 
                return False
    return len(stack) == 0

test = "(())"
print(is_valid_bracket(test))