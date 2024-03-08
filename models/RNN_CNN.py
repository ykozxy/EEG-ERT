import torch
from optuna import Trial
from torch import nn


class RNNWithCNN(nn.Module):
    """
    Combination of RNN and CNN.

    Architecture
     - Input: (N, D, T), D (num of features), T (num of time steps)
    - 1D Convolutions along temporal dimension
        - Conv - BN - ReLU - MaxPool
     - RNN (LSTM)
     - Fully connected layers
    """

    def __init__(
        self,
        trial: Trial,
        num_class: int = 4,
        features_dim: int = 22,
        time_dim: int = 1000,
    ):
        super(RNNWithCNN, self).__init__()

        # CNN
        num_cnn_layers = trial.suggest_int("num_cnn_layers", 3, 10)
        cnn_use_bn = trial.suggest_categorical("cnn_use_bn", [True, False])
        cnn_activation = trial.suggest_categorical("cnn_activation", ["ReLU", "ELU"])
        cnn_dropout = trial.suggest_float("cnn_dropout", 0.0, 0.5)
        kernel_size = trial.suggest_int(f"cnn_kernel_size", 3, 20)

        cnn_layers = nn.ModuleList()
        for i in range(num_cnn_layers):
            out_channels = trial.suggest_int(f"cnn_out_channels_{i}", 2, 200)
            padding = kernel_size // 2
            cnn_layers.append(
                nn.LazyConv1d(
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )

            if cnn_activation == "ReLU":
                cnn_layers.append(nn.ReLU())
            else:
                cnn_layers.append(nn.ELU())

            if cnn_use_bn:
                cnn_layers.append(nn.LazyBatchNorm1d())

            cnn_layers.append(nn.MaxPool1d(kernel_size=2))

            cnn_layers.append(nn.Dropout(cnn_dropout))

        self.cnn = nn.Sequential(*cnn_layers)

        # Calculate CNN output size
        test_tensor = torch.randn(1, features_dim, time_dim)
        with torch.no_grad():
            test_tensor = self.cnn(test_tensor)
        rnn_in_size = test_tensor.shape[2]

        # Fully connected layers - 1
        self.use_fc1 = trial.suggest_categorical("use_fc1", [True, False])
        if self.use_fc1:
            fc_out_size = trial.suggest_int("rnn_in_size", 64, 700)
            self.fc1 = nn.Sequential(
                nn.Flatten(),
                nn.LazyLinear(fc_out_size),
            )
            rnn_in_size = 1

        # RNN
        num_rnn_layers = trial.suggest_int("num_rnn_layers", 1, 8)
        rnn_hidden_size = trial.suggest_int("rnn_hidden_size", 16, 512)
        rnn_dropout = trial.suggest_float("rnn_dropout", 0.0, 0.5)
        self.rnn = nn.LSTM(
            input_size=rnn_in_size,
            hidden_size=rnn_hidden_size,
            num_layers=num_rnn_layers,
            dropout=rnn_dropout,
            batch_first=True,
        )

        # Fully connected layers - 2
        num_fc_layers = trial.suggest_int("num_fc_layers_2", 0, 3)
        fc_use_bn = trial.suggest_categorical("fc_use_bn_1", [True, False])
        fc_layers_2 = nn.ModuleList()
        fc_layers_2.append(nn.Flatten())
        for i in range(num_fc_layers):
            out_features = trial.suggest_int(f"fc_out_features_1_{i}", 16, 1024)
            dropout = trial.suggest_float(f"fc_dropout_1_{i}", 0.0, 0.5)
            fc_layers_2.append(nn.LazyLinear(out_features))
            if fc_use_bn:
                fc_layers_2.append(nn.BatchNorm1d(out_features))
            fc_layers_2.append(nn.ReLU())
            fc_layers_2.append(nn.Dropout(dropout))

        # Output layer
        fc_layers_2.append(nn.LazyLinear(num_class))
        self.fc2 = nn.Sequential(*fc_layers_2)

    def forward(self, x):
        # x = x.permute(0, 2, 1)

        x = self.cnn(x)

        if self.use_fc1:
            x = self.fc1(x)
            x = x.unsqueeze(2)

        x, _ = self.rnn(x)
        x = self.fc2(x)

        # Softmax
        softmax = nn.Softmax(dim=1)
        x = softmax(x)

        return x
