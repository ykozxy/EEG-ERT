import numpy as np
import torch


class EEGDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for EEG data"""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        shuffle: bool = True,
        label_smoothing: bool = True,
    ):
        self.X = X
        self.y = y

        # Check shape
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("X and y must have the same length")

        # Encode y
        self.y -= self.y.min()
        self.num_class = self.y.max() + 1
        if label_smoothing:
            # True class probability: 0.9
            # Other class probability: 0.1 / (num_class - 1)
            self.y = np.eye(self.num_class)[self.y] * 0.9
            self.y[self.y == 0] = 0.1 / (self.num_class - 1)
        else:
            self.y = np.eye(self.num_class)[self.y]

        if shuffle:
            idx = np.random.permutation(self.X.shape[0])
            self.X = self.X[idx]
            self.y = self.y[idx]

        # Convert to float32
        self.X = self.X.astype(np.float32)
        self.y = self.y.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


if __name__ == "__main__":
    # Testing
    X_file = "../data/X_train_valid.npy"
    y_file = "../data/y_train_valid.npy"
    dataset = EEGDataset(np.load(X_file), np.load(y_file))
    print(len(dataset))
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
    print(dataset[0])
