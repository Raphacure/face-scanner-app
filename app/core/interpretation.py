def interpret_range(value, ranges):
    for label, (low, high) in ranges.items():
        if low <= value <= high:
            return label
    return "Unknown"
