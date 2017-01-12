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
from sklearn.metrics import accuracy_score
import toolbox

DIRNAME = "C:/Users/Kaur/Documents/CompNeuro/Project/idata/Dog_2"
os.chdir(DIRNAME)
subject = 2
preictals = 42
interictals = 120
ch1 = 10
ch2 = 14
ch3 = 6
ch4 = 5

"""
file = sp.loadmat('Dog_2_interictal_segment_0001.mat')
print(file.keys())
#print(file)
print(file['interictal_segment_1'][0][0]['sequence'])
"""

print('adding preictal segment data')
# loop 0
file = sp.loadmat('Dog_{:1d}_preictal_segment_{:04d}.mat'.format(subject, 1))
data1 = np.array([toolbox.extaggregate(toolbox.schaggregate(file['preictal_segment_' + str(1)][0][0]['data'], ch1))])
data2 = np.array([toolbox.extaggregate(toolbox.schaggregate(file['preictal_segment_' + str(1)][0][0]['data'], ch2))])
data3 = np.array([toolbox.extaggregate(toolbox.schaggregate(file['preictal_segment_' + str(1)][0][0]['data'], ch3))])
data4 = np.array([toolbox.extaggregate(toolbox.schaggregate(file['preictal_segment_' + str(1)][0][0]['data'], ch4))])
for i in range(1, preictals):
    file = sp.loadmat('Dog_{:1d}_preictal_segment_{:04d}.mat'.format(subject, i + 1))
    data1 = np.append(data1, np.array(
        [toolbox.extaggregate(toolbox.schaggregate(file['preictal_segment_' + str(i + 1)][0][0]['data'], ch1))]),
                      axis=0)
    data2 = np.append(data2, np.array(
        [toolbox.extaggregate(toolbox.schaggregate(file['preictal_segment_' + str(i + 1)][0][0]['data'], ch2))]),
                      axis=0)
    data3 = np.append(data3, np.array(
        [toolbox.extaggregate(toolbox.schaggregate(file['preictal_segment_' + str(i + 1)][0][0]['data'], ch3))]),
                      axis=0)
    data4 = np.append(data4, np.array(
        [toolbox.extaggregate(toolbox.schaggregate(file['preictal_segment_' + str(i + 1)][0][0]['data'], ch4))]),
                      axis=0)
    # print(np.array([file['preictal_segment_'+str(i+1)][0][0]['data']]).shape)
    # print('ictal:',i+1)
print(data1.shape)
print('adding interictal segment data')
for i in range(interictals):  # 100):#480):
    file = sp.loadmat('Dog_{:1d}_interictal_segment_{:04d}.mat'.format(subject, i + 1))
    data1 = np.append(data1, np.array(
        [toolbox.extaggregate(toolbox.schaggregate(file['interictal_segment_' + str(i + 1)][0][0]['data'], ch1))]),
                      axis=0)
    data2 = np.append(data2, np.array(
        [toolbox.extaggregate(toolbox.schaggregate(file['interictal_segment_' + str(i + 1)][0][0]['data'], ch2))]),
                      axis=0)
    data3 = np.append(data3, np.array(
        [toolbox.extaggregate(toolbox.schaggregate(file['interictal_segment_' + str(i + 1)][0][0]['data'], ch3))]),
                      axis=0)
    data4 = np.append(data4, np.array(
        [toolbox.extaggregate(toolbox.schaggregate(file['interictal_segment_' + str(i + 1)][0][0]['data'], ch4))]),
                      axis=0)
    # orint(np.array([file['interictal_segment_'+str(i+1)][0][0]['data']]).shape)
    # print('inter: ',i+1)
print(data1.shape)

data1 = data1.reshape((preictals + interictals,
                       data1.shape[1] * data1.shape[2]))  # make the data edible for classifiers expecting 2D array
data2 = data2.reshape((preictals + interictals, data2.shape[1] * data2.shape[2]))
data3 = data3.reshape((preictals + interictals, data3.shape[1] * data3.shape[2]))
data4 = data4.reshape((preictals + interictals, data4.shape[1] * data4.shape[2]))
print('data added')
# create list of results: 0 means no seizure, 1 means seizure in 55-65 minutes
# 2 means seizure in 45-55 minutes etc. up to 6 (5-15 minutes)
print('adding target data')

target = np.zeros(preictals + interictals)
for i in range(int(preictals / 6)):
    for j in range(6):  # seizure or not
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
clfK = KNeighborsClassifier(20)
# Random Forest
clfRF = RandomForestClassifier(max_depth=10, n_estimators=10, max_features=3)


# results:
def score(clf, name):
    score = cross_val_score(clf, data1, target, cv=3)
    print(name, score.mean(), '+-', score.std())
    score = cross_val_predict(clf, data1, target, cv=3)
    print(score)


def checkNB(clf, name):
    score1 = cross_val_predict(clf, data1, target, cv=3)
    score2 = cross_val_predict(clf, data2, target, cv=3)
    score3 = cross_val_predict(clf, data3, target, cv=3)
    score4 = cross_val_predict(clf, data4, target, cv=3)
    score = np.zeros(target.shape)
    for i in range(score.shape[0]):
        if score1[i] == score2[i] and score2[i] == score3[i] and score3[i] == score4[i] and score1[i] == 1:
            score[i] = 1
    print(accuracy_score(score, target))
    print(score)


print(target)
checkNB(clfNB, "Naive Bayes")

# score(clfNB,"Naive Bayes")
# score(clflSVC,"Linear SVC")
score(clfeSVC, "Sigmoid SVC")
# score(clfK,"K nearest")
# score(clfRF,"Random Forest")
