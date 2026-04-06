def delete_space(lines: list[str])-> list[str]:
    result=[]
    for line in lines:
        if line != "":
            result.append(line)
    return result

lines = ["hello", "", "world", "", "", "python"]
print(delete_space(lines))
