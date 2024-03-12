import numpy as np
from torch.utils.data import DataLoader

from prepare_frequency_data import prepare_frequency_data
from utils.dataset import EEGDataset
from braindecode.augmentation import AugmentedDataLoader
from braindecode.augmentation import FTSurrogate


def get_raw_dataloader(label_smoothing: bool = True, batch_size: int = 64):
    """
    Get dataloader for raw EEG data on all subjects
    """
    X_train_data = np.load("data/X_train_valid.npy")
    y_train_data = np.load("data/y_train_valid.npy")
    X_test_data = np.load("data/X_test.npy")
    y_test_data = np.load("data/y_test.npy")

    split_idx = 1777
    X_train_data, X_valid_data = X_train_data[:split_idx], X_train_data[split_idx:]
    y_train_data, y_valid_data = y_train_data[:split_idx], y_train_data[split_idx:]

    train_set = EEGDataset(X_train_data, y_train_data, label_smoothing=label_smoothing)
    valid_set = EEGDataset(X_valid_data, y_valid_data, label_smoothing=False)
    test_set = EEGDataset(X_test_data, y_test_data, label_smoothing=False)

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(valid_set, batch_size=batch_size, shuffle=False),
        DataLoader(test_set, batch_size=batch_size, shuffle=False),
    )


def get_sub1_dataloader(label_smoothing: bool = True, batch_size: int = 64):
    """
    Get dataloader for raw EEG data on subject 1
    """
    X_train_data = np.load("data/X_train_valid.npy")
    y_train_data = np.load("data/y_train_valid.npy")
    X_test_data = np.load("data/X_test.npy")
    y_test_data = np.load("data/y_test.npy")

    X_train_data, X_valid_data = X_train_data[:199], X_train_data[1777:1815]
    y_train_data, y_valid_data = y_train_data[:199], y_train_data[1777:1815]

    test_person = np.load("data/person_test.npy")
    sub0_index = (test_person == 0).reshape(-1)
    X_test_data = X_test_data[sub0_index]
    y_test_data = y_test_data[sub0_index]

    train_set = EEGDataset(X_train_data, y_train_data, label_smoothing=label_smoothing)
    valid_set = EEGDataset(X_valid_data, y_valid_data, label_smoothing=False)
    test_set = EEGDataset(X_test_data, y_test_data, label_smoothing=False)

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(valid_set, batch_size=batch_size, shuffle=False),
        DataLoader(test_set, batch_size=batch_size, shuffle=False),
    )


def get_frequency_dataloader(label_smoothing: bool = True, batch_size: int = 64):
    """
    Get dataloader for frequency domain EEG data on all subjects
    Training set is augmented by frequency filtering and FTSurroagte method
    proposed in https://arxiv.org/abs/1806.08675
    and experimented in https://arxiv.org/pdf/2206.14483.pdf
    Validation and test set are un-augmented raw data.
    """
    # prepare_frequency_data()

    # X_train_freq = np.load("data/generated/frequency/X_train_augmented.npy")
    # y_train_freq = np.load("data/generated/frequency/y_train_augmented.npy")
    X_train_raw = np.load("data/X_train_valid.npy")
    y_train_raw = np.load("data/y_train_valid.npy")
    X_test_data = np.load("data/X_test.npy")
    y_test_data = np.load("data/y_test.npy")

    split_idx = 1777
    X_train_data, X_valid_data = X_train_raw[:split_idx], X_train_raw[split_idx:]
    y_train_data, y_valid_data = y_train_raw[:split_idx], y_train_raw[split_idx:]

    train_set = EEGDataset(X_train_data, y_train_data, label_smoothing=label_smoothing)
    valid_set = EEGDataset(X_valid_data, y_valid_data, label_smoothing=False)
    test_set = EEGDataset(X_test_data, y_test_data, label_smoothing=False)
    ft_sr = FTSurrogate(probability=0.5, phase_noise_magnitude=0.9, channel_indep=False)

    return (
        AugmentedDataLoader(train_set, transforms=[ft_sr], batch_size=batch_size, shuffle=True),
        DataLoader(valid_set, batch_size=batch_size, shuffle=False),
        DataLoader(test_set, batch_size=batch_size, shuffle=False),
    )


def get_time_series_dataloader(time_stamp: int, label_smoothing: bool = True, batch_size: int = 64):
    """
    Get dataloader for time series data on all subjects
    """
    X_train_data = np.load("data/X_train_valid.npy")
    y_train_data = np.load("data/y_train_valid.npy")
    X_test_data = np.load("data/X_test.npy")
    y_test_data = np.load("data/y_test.npy")

    X_train_data = X_train_data[:, :, :time_stamp]
    X_test_data = X_test_data[:, :, :time_stamp]

    split_idx = 1777
    X_train_data, X_valid_data = X_train_data[:split_idx], X_train_data[split_idx:]
    y_train_data, y_valid_data = y_train_data[:split_idx], y_train_data[split_idx:]

    train_set = EEGDataset(X_train_data, y_train_data, label_smoothing=label_smoothing)
    valid_set = EEGDataset(X_valid_data, y_valid_data, label_smoothing=False)
    test_set = EEGDataset(X_test_data, y_test_data, label_smoothing=False)

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(valid_set, batch_size=batch_size, shuffle=False),
        DataLoader(test_set, batch_size=batch_size, shuffle=False),
    )
