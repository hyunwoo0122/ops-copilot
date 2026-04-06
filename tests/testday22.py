# 배열을 인자로 받는 함수 만들기
# 딕셔너리로 나오는 걸 저장하면서 횟수를 적어

def process_ids(id_list: list[str]) -> str | None:
    result_dict = {}

    for code in id_list:
        if code in result_dict:
            result_dict[code] += 1
            if result_dict[code] == 2:
                return code
        else:
            result_dict[code] = 1
    return None


data = ["E01", "E02", "E01", "E03", "E02", "E02"]
result = process_ids(data)
print("처음으로 두 번 나온 에러 코드:", result)

