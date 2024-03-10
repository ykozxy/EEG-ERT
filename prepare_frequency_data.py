import os
import numpy as np
import mne
import braindecode.augmentation as Augmentations

os.makedirs('data/generated/frequency', exist_ok=True)

sfreq = 250

print(f"Loading data...")

X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")
person_train_valid = np.load("data/person_train_valid.npy")
X_train_valid = np.load("data/X_train_valid.npy")
y_train_valid = np.load("data/y_train_valid.npy")
person_test = np.load("data/person_test.npy")

X_train = X_train_valid[:1777]
y_train = y_train_valid[:1777]
person_train = person_train_valid[:1777]

X_train_aug = [X_train]
concat_times = len(X_train_aug)

# print("filtering data...")
# X_train_filtered_30_100 = [mne.filter.filter_data(trial, sfreq, 30, 100, verbose=False) for trial in X_train]
# X_train_filtered_0_30 = [mne.filter.filter_data(trial, sfreq, 0, 30, verbose=False) for trial in X_train]
# X_test_filtered_30_100 = [mne.filter.filter_data(trial, sfreq, 30, 100, verbose=False) for trial in X_test]
# X_test_filtered_0_30 = [mne.filter.filter_data(trial, sfreq, 0, 30, verbose=False) for trial in X_test]

surrogate_times = 4
for i in range(surrogate_times):
    print(f"generating ft surrogate ({i})...")
    ftsurrogate = Augmentations.FTSurrogate(probability=0.5, channel_indep=False)
    sr = np.array([ftsurrogate(trial) for trial in X_train])
    X_train_aug.append(sr)
    concat_times += 1

stm_times = 0
for i in range(stm_times):
    print(f"generating smooth time masks ({i})...")
    smooth = Augmentations.SmoothTimeMask(probability=1, mask_len_samples=140)
    stm = np.array([smooth(trial) for trial in X_train])
    X_train_aug.append(stm)
    concat_times += 1

print("concatenating...")
# X_train_augmented = np.concatenate([X_train, X_train_filtered_30_100, X_train_filtered_0_30])
X_train_augmented = np.concatenate(X_train_aug)
y_train_augmented = np.concatenate([y_train] * (concat_times))
person_train_augmented = np.concatenate([person_train] * (concat_times))

print("saving...")
np.save("data/generated/frequency/X_train_augmented.npy", X_train_augmented)
np.save("data/generated/frequency/person_train_augmented.npy", person_train_augmented)
# np.save("data/generated/frequency/X_test_filtered_30_100.npy", X_test_filtered_30_100)
# np.save("data/generated/frequency/X_test_filtered_0_30.npy", X_test_filtered_0_30)
np.save("data/generated/frequency/y_train_augmented.npy", y_train_augmented)

print("done!")