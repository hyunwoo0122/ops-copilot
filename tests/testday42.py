def sorted_log(logs: list[str])->list[str]:
    return sorted(set(logs))

logs = ["ERROR", "INFO", "WARNING", "ERROR", "INFO", "CRITICAL"]
print(sorted_log(logs))