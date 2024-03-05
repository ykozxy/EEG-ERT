import os
import numpy as np
import mne

os.makedirs('data/generated/frequency', exist_ok=True)

sfreq = 250

X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")
person_train_valid = np.load("data/person_train_valid.npy")
X_train_valid = np.load("data/X_train_valid.npy")
y_train_valid = np.load("data/y_train_valid.npy")
person_test = np.load("data/person_test.npy")

print("filtering data...")
X_train_valid_filtered_30_100 = [mne.filter.filter_data(trial, sfreq, 30, 100, verbose=False) for trial in X_train_valid]
X_train_valid_filtered_0_30 = [mne.filter.filter_data(trial, sfreq, 0, 30, verbose=False) for trial in X_train_valid]
X_test_filtered_30_100 = [mne.filter.filter_data(trial, sfreq, 30, 100, verbose=False) for trial in X_test]
X_test_filtered_0_30 = [mne.filter.filter_data(trial, sfreq, 0, 30, verbose=False) for trial in X_test]

print("concatenating...")
X_train_valid_augmented = np.concatenate([X_train_valid, X_train_valid_filtered_30_100, X_train_valid_filtered_0_30])
y_train_valid_augmented = np.concatenate([y_train_valid] * 3)
person_train_valid_augmented = np.concatenate([person_train_valid] * 3)

print("saving...")
np.save("data/generated/frequency/X_train_valid_augmented.npy", X_train_valid_augmented)
np.save("data/generated/frequency/person_train_valid_augmented.npy", person_train_valid_augmented)
np.save("data/generated/frequency/X_test_filtered_30_100.npy", X_test_filtered_30_100)
np.save("data/generated/frequency/X_test_filtered_0_30.npy", X_test_filtered_0_30)
np.save("data/generated/frequency/y_train_valid_augmented.npy", y_train_valid_augmented)

print("done!")