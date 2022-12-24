import numpy as np


# one-hot编码
def one_hot(data: np.ndarray, num_classes: int):
    """Convert an iterable of indices to one-hot encoded labels."""
    return np.eye(num_classes)[data]


# 标准化向量
def standardize(x: np.ndarray):
    """ Standardize data """
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)


# 归一化向量
def Normalize(data: np.ndarray):
    """ Normalize data """
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))


# Test
if __name__ == '__main__':
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(standardize(a.T).T)
    print(Normalize(a.T).T)

    print("-" * 100)

    label = np.array([0, 1, 2])
    print(label)
    print(one_hot(label, 3))
