import numpy as np

X_train_valid = np.load("data/X_train_valid.npy")
y_train_valid = np.load("data/y_train_valid.npy")

X_sub1_train = X_train_valid[:199]
y_sub1_train = y_train_valid[:199]
X_sub1_valid = X_train_valid[1777:1815]
y_sub1_valid = y_train_valid[1777:1815]