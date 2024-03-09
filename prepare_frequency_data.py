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

X_train = X_train_valid[:1777]
y_train = y_train_valid[:1777]
person_train = person_train_valid[:1777]

X_train_aug = [X_train]

print("filtering data...")
X_train_filtered_30_100 = [mne.filter.filter_data(trial, sfreq, 30, 100, verbose=False) for trial in X_train]
X_train_filtered_0_30 = [mne.filter.filter_data(trial, sfreq, 0, 30, verbose=False) for trial in X_train]
X_test_filtered_30_100 = [mne.filter.filter_data(trial, sfreq, 30, 100, verbose=False) for trial in X_test]
X_test_filtered_0_30 = [mne.filter.filter_data(trial, sfreq, 0, 30, verbose=False) for trial in X_test]

surrogate_times = 5
for i in range(surrogate_times):
    print(f"transforming data ({i})...")
    fft_data = np.fft.fft(X_train, axis=-1)
    random_phase = np.exp(1j * np.random.uniform(0, 2 * np.pi, size=fft_data.shape))
    fft_data_randomized = fft_data * random_phase
    data_randomized = np.fft.ifft(fft_data_randomized, axis=-1)
    data_randomized = np.real(data_randomized)
    X_train_aug.append(data_randomized)

print("concatenating...")
# X_train_augmented = np.concatenate([X_train, X_train_filtered_30_100, X_train_filtered_0_30])
X_train_augmented = np.concatenate(X_train_aug)
y_train_augmented = np.concatenate([y_train] * (1 + surrogate_times))
person_train_augmented = np.concatenate([person_train] * (1 + surrogate_times))

print("saving...")
np.save("data/generated/frequency/X_train_augmented.npy", X_train_augmented)
np.save("data/generated/frequency/person_train_augmented.npy", person_train_augmented)
# np.save("data/generated/frequency/X_test_filtered_30_100.npy", X_test_filtered_30_100)
# np.save("data/generated/frequency/X_test_filtered_0_30.npy", X_test_filtered_0_30)
np.save("data/generated/frequency/y_train_augmented.npy", y_train_augmented)

print("done!")