import numpy as np


def normalized(data):
    """
    [batch, dim, channel]
    """
    assert len(data.shape) == 3
    batch, dim, channel = data.shape
    for i in range(batch):
        for j in range(channel):
            if np.std(data[i, :, j]) != 0:
                data[i, :, j] = (data[i, :, j] - np.mean(data[i, :, j])) / np.std(data[i, :, j])
            else:
                data[i, :, j] = 0
    return data


def standardize(data):
    """
    [batch, dim, channel]
    """
    assert len(data.shape) == 3
    batch, dim, channel = data.shape
    for i in range(batch):
        for j in range(channel):
            if np.max(data[i, :, j]) - np.min(data[i, :, j]) != 0:
                data[i, :, j] = (data[i, :, j] - np.min(data[i, :, j])) / \
                                (np.max(data[i, :, j]) - np.min(data[i, :, j] + 1e-8))
            else:
                data[i, :, j] = 0
    return data


def oneHot(label, class_num):
    """
    [batch,]
    """
    assert len(label.shape) == 1
    label = label.astype(np.int32)
    batch = label.shape[0]
    label = label.reshape(batch)
    label = np.eye(class_num)[label]
    return label


if __name__ == '__main__':
    a = np.random.random((2, 2, 2))
    print(a)
    print(normalized(a))
    print(standardize(a))
    b = np.array([0, 1, 2, 3, 4, 5])
    print(oneHot(b, 6))
