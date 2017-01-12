import numpy as np


def aggregate(arr):
    x = arr.shape[0]
    y = int(arr.shape[1] / 3)
    out = np.empty((x, y))
    for i in range(x):
        for j in range(y):
            out[i][j] = (arr[i][j * 3] + arr[i][j * 3 + 1] + arr[i][j * 3 + 2]) / 3
    return out


def extaggregate(arr):
    x = arr.shape[0]
    if x > 16:
        y = int(arr.shape[0] / 100)
        x = 1
        out = np.empty((x, y))
        for j in range(y):
            sum = 0
            for m in range(100):
                sum = arr[j * 100 + m]
            out[0][j] = sum / 100
    else:
        y = int(arr.shape[1] / 100)
        out = np.empty((x, y))
        for i in range(x):
            for j in range(y):
                sum = 0
                for m in range(100):
                    sum = arr[i][j * 100 + m]
                out[i][j] = sum / 100
    return out


def chaggregate(arr):
    x = 4
    y = arr.shape[1]
    out = np.empty((x, y))
    for i in range(x):
        for j in range(y):
            if i == 0:
                out[i][j] = arr[1][j]
            else:
                out[i][j] = arr[i + 7][j]
    return out


def schaggregate(arr, ch):
    return arr[ch - 1]
