import numpy as np


class EEGDataset:
    """ PyTorch Dataset for EEG data """

    def __init__(self, X_file: str, y_file: str, shuffle: bool = True):
        self.X = np.load(X_file)
        self.y = np.load(y_file)

        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("X and y must have the same length")

        if shuffle:
            idx = np.random.permutation(self.X.shape[0])
            self.X = self.X[idx]
            self.y = self.y[idx]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


if __name__ == '__main__':
    # Testing
    X_file = '../data/X_train_valid.npy'
    y_file = '../data/y_train_valid.npy'
    dataset = EEGDataset(X_file, y_file)
    print(len(dataset))
    print(dataset[0][0].shape)
    print(dataset[0][1])
