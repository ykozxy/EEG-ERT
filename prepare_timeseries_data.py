# Generate training/testing data with 250/500/750 time steps

import os

import numpy as np

os.makedirs('data/generated/time_series', exist_ok=True)

x_test = np.load('data/x_test.npy')
x_train = np.load('data/x_train_valid.npy')

time_steps = [250, 500, 750]

for t in time_steps:
    x_train_t = x_train[:, :, :t]
    np.save('data/generated/time_series/x_train_valid_{}.npy'.format(t), x_train_t)

    x_test_t = x_test[:, :, :t]
    np.save('data/generated/time_series/x_test_{}.npy'.format(t), x_test_t)
