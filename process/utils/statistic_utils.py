import numpy as np

def normalize_to_0_1(data):
    data = np.array(data)
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val - min_val == 0:
        return data # 防止除以零
    return np.array([(x - min_val) / (max_val - min_val) for x in data])