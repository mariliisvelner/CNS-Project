from numpy import *
import os
import numpy as np

from sklearn import svm, metrics
from sklearn.cross_validation import cross_val_score, cross_val_predict

DATA_DIR = "<directory of the folder Dog_1_processed>"
# The data is in the Google Docs shared folder

os.chdir(DATA_DIR)

"""
string filename - the name of a .dat file to be processed

Returns the matrix, which contains the data in the given file.
"""


def get_data(filename):
    file = open(filename, 'rb')

    m = np.asscalar(fromfile(file, uint64, 1))
    n = np.asscalar(fromfile(file, uint64, 1))

    # print(m)
    # print(n)

    x = zeros((m, n + 2))

    for i in range(m):
        x[i] = fromfile(file, double, n + 2)
        # print(type(x[i])) # x[i] is a tuple with a size of 282

    file.close()

    # print(x.shape)
    # print(type(x))
    return x


# ICTAL - 1. column is 1
# INTER - 1. column is 2

N = 10000
data_ict = get_data("dog1.ictal.dat")
data_int = get_data("dog1.inter.dat")[:N, :]  # takes the first N rows of the data

# print(x[:, 2:])

data = np.concatenate((data_ict, data_int), axis=0)
# The first column of the matrix contains the labels we need to predict (1 for preictal, 2 for interictal)
target = data[:, 0]
data = data[:, 1:]

clf = svm.SVC(kernel='linear', C=1)
# clf = svm.SVC(kernel='rbf', C=1)
# clf = svm.SVC(kernel='poly', C=1)
# clf = svm.SVC(kernel='sigmoid', C=1)
# clf = svm.SVC(kernel='precomputed', C=1)
# clf = neighbors.KNeighborsClassifier()

predictions = cross_val_predict(clf, data, target, cv=5)
print(np.sum(predictions == 1))
print(np.sum(predictions == 2))
score = metrics.accuracy_score(target, predictions)
print(score)
scores = cross_val_score(clf, data, target, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
