def f1(precision, recall):
    return 2 * precision * recall / (precision + recall)

print("Qwen2.5-VL-72B")
precision = 0.90
recall = 0.69
print(f1(precision, recall))

print("Gemini-2.5-Pro")
precision = 0.81
recall = 0.98
print(f1(precision, recall))