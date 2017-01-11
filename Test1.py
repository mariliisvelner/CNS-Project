from numpy import *
import os
import numpy as np

from sklearn import svm
from sklearn.cross_validation import cross_val_score

DATA_DIR = "<directory of Dog_1_processed>"

os.chdir(DATA_DIR)


def get_data(filename):
    file = open(filename, 'rb')

    m = np.asscalar(fromfile(file, uint64, 1))
    n = np.asscalar(fromfile(file, uint64, 1))

    print(m)
    print(n)

    x = zeros((m, n + 2))

    for i in range(m):
        x[i] = fromfile(file, double, n + 2)
        # print(type(x[i])) # x[i] is a tuple with a size of 282

    file.close()

    print(x.shape)
    print(type(x))
    return x

# ICTAL - 1. column is 1
# INTER - 1. column is 2
# Big question -- what do the numbers in the second column represent?


N = 1000 # TODO: change this
data_ict = get_data("dog1.ictal.dat")
data_int = get_data("dog1.inter.dat")[:N, :] # takes the first N rows of the data

# print(x[:, 2:])

data = np.concatenate((data_ict, data_int), axis=0)
target = data[:, 0]
data = data[:, 1:]
# print(data.shape)
# print(target.shape)

clf = svm.SVC(kernel='linear', C=1)
# clf = svm.SVC(kernel='rbf', C=1)
# clf = svm.SVC(kernel='poly', C=1)
# clf = svm.SVC(kernel='sigmoid', C=1)
# clf = svm.SVC(kernel='precomputed', C=1)
scores = cross_val_score(clf, data, target, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
