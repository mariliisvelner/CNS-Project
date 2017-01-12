from numpy import *
import os 
import numpy as np
import pandas as pd

os.chdir("/Users/krister.jaanhold/Downloads")

file = open("dog1.inter.dat", 'rb')

m = np.asscalar(fromfile(file, uint64, 1))
n = np.asscalar(fromfile(file, uint64, 1))

X = zeros((m, n + 2))

for i in range(m):
    X[i] = fromfile(file, double, n + 2)

file.close()

data = pd.DataFrame(X)

data.to_csv('dog1.inter.csv', index=False, header=False)

file = open("dog1.ictal.dat", 'rb')

m = np.asscalar(fromfile(file, uint64, 1))
n = np.asscalar(fromfile(file, uint64, 1))

X = zeros((m, n + 2))

for i in range(m):
    X[i] = fromfile(file, double, n + 2)

file.close()

data = pd.DataFrame(X)

data.to_csv('dog1.ictal.csv', index=False, header=False)

file = open("dog1.test.dat", 'rb')

m = np.asscalar(fromfile(file, uint64, 1))
n = np.asscalar(fromfile(file, uint64, 1))

X = zeros((m, n + 2))

for i in range(m):
    X[i] = fromfile(file, double, n + 2)

file.close()

data = pd.DataFrame(X)

data.to_csv('dog1.test.csv', index=False, header=False)