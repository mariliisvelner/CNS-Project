from numpy import *
import os
import numpy as np
import scipy.io as sp
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import random

DATA_DIR = "<directory of Dog_X>"
DOG = "X"

os.chdir(DATA_DIR)

"""
file = sp.loadmat('Dog_1_interictal_segment_0001.mat')
print(file.keys())
#print(file)
print(file['interictal_segment_1'][0][0]['sequence'])
"""

# create list of files to use
filelist = []
# data=np.array([])
print('adding preictal segment data')
# loop 0
filelist.append('Dog_{}_preictal_segment_{:04d}.mat'.format(DOG, 1))
file = sp.loadmat('Dog_{}_preictal_segment_{:04d}.mat'.format(DOG, 1))
data = np.array([file['preictal_segment_' + str(1)][0][0]['data']])
for i in range(1, 24):
    filelist.append('Dog_{}_preictal_segment_{:04d}.mat'.format(DOG, i + 1))
    file = sp.loadmat('Dog_{}_preictal_segment_{:04d}.mat'.format(DOG, i + 1))
    data = np.append(data, np.array([file['preictal_segment_' + str(i + 1)][0][0]['data']]), axis=0)
    # print(np.array([file['preictal_segment_'+str(i+1)][0][0]['data']]).shape)
    # print('ictal:',i+1)
print(data.shape)
print('adding interictal segment data')
for i in range(100):  # 480):
    filelist.append('Dog_{}_interictal_segment_{:04d}.mat'.format(DOG, i + 1))
    file = sp.loadmat('Dog_{}_interictal_segment_{:04d}.mat'.format(DOG, i + 1))
    data = np.append(data, np.array([file['interictal_segment_' + str(i + 1)][0][0]['data']]), axis=0)
    # orint(np.array([file['interictal_segment_'+str(i+1)][0][0]['data']]).shape)
    # print('inter: ',i+1)
print(data.shape)
x = data.shape[0] # Dog 5: 124
y = data.shape[1] * data.shape[2]

data = data.reshape((x, y))  # make the data edible for classifiers expecting 2D array
print('data added')
# create list of results: 0 means no seizure, 1 means seizure
print('adding target data')
"""
target=np.array([])
for i in range(4):
    for j in range(6):
        np.append(target,j)
        #this way we can not only predict whether there will be a seizure,
        #but also when the seizure will take place
for i in range(100):#480):
    np.append(target,0)
    """
target = np.zeros(124)
for i in range(4):
    for j in range(6):  # seizure oncoming
        target[i + j] = 1

print('target data added')
print(target.shape)

# Data added, time to divide it for learning and testing
print('making predictions')
# Naive Bayes
clfNB = GaussianNB()
# linear Support Vector Classifier
clflSVC = svm.SVC(kernel='linear', C=1)
# ensemble SVC
clfeSVC = svm.SVC(kernel='sigmoid', C=1)
# K-nearest
clfK = KNeighborsClassifier(4)
# Random Forest
clfRF = RandomForestClassifier(max_depth=10, n_estimators=10, max_features=3)


# results:
def score(clf, name):
    score = cross_val_score(clf, data, target, cv=3)
    print(name, score.mean(), '+-', score.std())


score(clfNB, "Naive Bayes")
score(clflSVC, "Linear SVC")
score(clfeSVC, "Ensemble SVC")
score(clfK, "K nearest")
score(clfRF, "Random Forest")
