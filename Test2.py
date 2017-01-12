from numpy import *
import os
import numpy as np
import scipy.io as sp
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import toolbox

DIRNAME = "C:/Users/Kaur/Documents/CompNeuro/Project/idata/Dog_2"
os.chdir(DIRNAME)
subject = 2
preictals = 42
interictals = 50

"""
file = sp.loadmat('Dog_2_interictal_segment_0001.mat')
print(file.keys())
#print(file)
print(file['interictal_segment_1'][0][0]['sequence'])
"""

print('adding preictal segment data')
# loop 0
file = sp.loadmat('Dog_{:1d}_preictal_segment_{:04d}.mat'.format(subject, 1))
data = np.array([toolbox.aggregate(file['preictal_segment_' + str(1)][0][0]['data'])])
for i in range(1, preictals):
    file = sp.loadmat('Dog_{:1d}_preictal_segment_{:04d}.mat'.format(subject, i + 1))
    data = np.append(data, np.array([toolbox.aggregate(file['preictal_segment_' + str(i + 1)][0][0]['data'])]), axis=0)
    # print(np.array([file['preictal_segment_'+str(i+1)][0][0]['data']]).shape)
    # print('ictal:',i+1)
print(data.shape)
print('adding interictal segment data')
for i in range(interictals):  # 100):#480):
    file = sp.loadmat('Dog_{:1d}_interictal_segment_{:04d}.mat'.format(subject, i + 1))
    data = np.append(data, np.array([toolbox.aggregate(file['interictal_segment_' + str(i + 1)][0][0]['data'])]),
                     axis=0)
    # orint(np.array([file['interictal_segment_'+str(i+1)][0][0]['data']]).shape)
    # print('inter: ',i+1)
print(data.shape)

data = data.reshape(
    (preictals + interictals, 1278752))  # 3836256)) #make the data edible for classifiers expecting 2D array
print('data added')
# create list of results: 0 means no seizure, 1 means seizure in 55-65 minutes
# 2 means seizure in 45-55 minutes etc. up to 6 (5-15 minutes)
print('adding target data')

target = np.zeros(preictals + interictals)
for i in range(int(preictals / 6)):
    for j in range(6):  # only knows seizure is coming
        target[i * 6 + j] = 1

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
    # score=cross_val_score(clf,data,target,cv=3)
    # print(name,score.mean(),'+-',score.std())
    score = cross_val_predict(clf, data, target, cv=3)
    print(score)


score(clfNB, "Naive Bayes")
score(clflSVC, "Linear SVC")
score(clfeSVC, "Sigmoid SVC")
score(clfK, "K nearest")
score(clfRF, "Random Forest")
