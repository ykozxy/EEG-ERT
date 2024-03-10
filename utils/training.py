import os
import pickle
from typing import Any, Callable, Tuple

import numpy as np
import optuna
import torch
from optuna import Trial, Study
from retry import retry
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.early_stopping import EarlyStopping


def best_torch_device():
    """Returns the best pytorch training device available"""
    if torch.backends.mps.is_available():
        # For Apple Silicon
        return torch.device("mps")
    elif torch.cuda.is_available():
        # For CUDA enabled GPUs
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_study_callback(cp_path: str):
    return lambda study, trial: optuna_study_callback(study, trial, cp_path)


def optuna_study_callback(
    study: Study,
    trial: Trial,
    cp_path: str,
):
    """
    Optuna callback function to save study and best model checkpoints
    :param study: Optuna study
    :param trial: Optuna trial
    :param cp_path: Checkpoint save directory
    """
    @retry(tries=5, delay=5, backoff=5, logger=None)
    def _remove_file(f):
        os.remove(f)

    study_name = study.study_name

    if not os.path.exists(cp_path):
        os.makedirs(cp_path, exist_ok=True)

    # Save study
    with open(f"{cp_path}/{study_name}_study.pkl", "wb") as f:
        pickle.dump(study, f)

    if len(study.trials) == 0:
        return

    # Rename to "best.pt" if best
    if study.best_trial is not None and trial.number == study.best_trial.number:
        if os.path.exists(f"{cp_path}/{study_name}_best.pt"):
            _remove_file(f"{cp_path}/{study_name}_best.pt")
        os.rename(
            f"{cp_path}/{study_name}_{trial.number}.pt",
            f"{cp_path}/{study_name}_best.pt",
        )
    elif os.path.exists(f"{cp_path}/{study_name}_{trial.number}.pt"):
        _remove_file(f"{cp_path}/{study_name}_{trial.number}.pt")


def optuna_train(
    trial: Trial,
    func_gen_model: Callable[[Trial], nn.Module],
    func_gen_optimizer: Callable[[Trial, nn.Module], Tuple[torch.optim.Optimizer, Any]],
    loss_func: nn.Module,
    train_data: DataLoader,
    val_data: DataLoader,
    cp_path: str = None,
    n_epochs: int = 100,
    show_progress_bar: bool = True,
    verbose: bool = True,
):
    """
    Optuna training routine for one trial.
    :param trial: Optuna trial object
    :param func_gen_model: Function that returns a pytorch model given a trial
    :param func_gen_optimizer: Function that returns a pytorch optimizer and an optional scheduler given a trial and model
    :param loss_func: Pytorch loss function
    :param train_data: Training set DataLoader. Each batch should be a tuple (X, y).
    :param val_data: Validation set DataLoader. Each batch should be a tuple (X, y).
    :param cp_path: Directory to save checkpoints. If None, uses "checkpoints/{trial.study.study_name}"
    :param n_epochs: Number of epochs to train. Default: 100
    :param show_progress_bar: Show progress bar. Default: True
    :param verbose: Print verbose logs. Default: True
    :return: Validation accuracy of the best model
    """
    if cp_path is None:
        cp_path = f"checkpoints/{trial.study.study_name}"

    if not os.path.exists(cp_path):
        os.makedirs(cp_path, exist_ok=True)

    device = best_torch_device()

    if verbose:
        print(f"\nTrial {trial.number}")
        print(f"Using device: {device}")

    # Generate model
    model = func_gen_model(trial).to(device)

    # Generate optimizer
    optimizer, scheduler = func_gen_optimizer(trial, model)

    # Print trial params
    if verbose:
        print(f"Params:")
        for key, value in trial.params.items():
            print(f"  {key}: {value}")

    # Setup early stopping
    early_stopping = EarlyStopping(
        verbose=verbose,
        path=f"{cp_path}/{trial.study.study_name}_{trial.number}.pt",
        delta=1e-6,
        patience=8,
    )

    val_loss_min = np.Inf
    val_acc_max = 0
    train_loss_hist = []
    train_acc_hist = []
    for epoch in range(n_epochs):
        model.train()

        train_loss = 0
        with tqdm(
            train_data, unit="batch", leave=False, disable=not show_progress_bar
        ) as t:
            t.set_description(f"Epoch {epoch}")

            # Train the model
            for i, (X, y) in enumerate(t):
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = loss_func(pred, y)
                train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                t.set_postfix(loss=f"{loss:.4f}")
            
            if scheduler is not None:
                scheduler.step()

        train_loss /= len(train_data.dataset)
        t.set_postfix(loss=f"{train_loss:.4f}")
        train_loss_hist.append(train_loss)

        # Evaluate training accuracy
        model.eval()
        train_acc = 0
        with torch.no_grad():
            for X, y in train_data:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                train_acc += (pred.argmax(dim=1) == y.argmax(dim=1)).sum().item()
        train_acc /= len(train_data.dataset)
        train_acc_hist.append(train_acc)

        # Evaluate validation loss and accuracy
        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for X, y in val_data:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                val_loss += loss_func(pred, y).item()
                val_acc += (pred.argmax(dim=1) == y.argmax(dim=1)).sum().item()
        val_loss /= len(val_data.dataset)
        val_acc /= len(val_data.dataset)
        if verbose:
            print(
                f"[Epoch {epoch}] val_loss={val_loss:.6f} val_acc={val_acc:.6f} train_loss={train_loss:.6f}"
            )

        val_loss_min = min(val_loss_min, val_loss)
        val_acc_max = max(val_acc_max, val_acc)

        # Early stopping
        model.to("cpu")  # Save model on CPU to avoid pytorch bugs
        early_stopping(val_loss, model)
        model.to(device)
        if early_stopping.early_stop:
            if verbose:
                print("Early stopping.")
            break

        # Optuna pruning
        trial.report(val_acc_max, step=epoch)
        if trial.should_prune():
            trial.set_user_attr("all_train_loss", train_loss_hist)
            trial.set_user_attr("all_train_acc", train_acc_hist)
            raise optuna.TrialPruned()

    trial.set_user_attr("all_train_loss", train_loss_hist)
    trial.set_user_attr("all_train_acc", train_acc_hist)

    return val_acc_max


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_func: nn.Module,
    train_data: DataLoader,
    val_data: DataLoader,
    cp_path: str = "checkpoints",
    cp_filename: str = "best.pt",
    n_epochs: int = 100,
):
    """
    Training routine without Optuna.
    :param model: Pytorch model
    :param optimizer: Pytorch optimizer
    :param loss_func: Pytorch loss function
    :param train_data: Training set DataLoader. Each batch should be a tuple (X, y).
    :param val_data: Validation set DataLoader. Each batch should be a tuple (X, y).
    :param cp_path: Directory to save checkpoints. Default: "checkpoints"
    :param cp_filename: Filename of the best model checkpoint. Default: "best.pt"
    :param n_epochs: Number of epochs to train. Default: 100
    :return: Validation loss of the best model
    """
    if not os.path.exists(cp_path):
        os.makedirs(cp_path, exist_ok=True)

    device = best_torch_device()
    print(f"Using device: {device}")

    # Setup early stopping
    early_stopping = EarlyStopping(verbose=True, path=f"{cp_path}/{cp_filename}")

    train_loss_hist = []
    val_loss_hist = []
    for epoch in range(n_epochs):
        model.train()

        train_loss = 0
        with tqdm(train_data, unit="batch", leave=False) as t:
            t.set_description(f"Epoch {epoch}")

            # Train the model
            for X, y in t:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = loss_func(pred, y)
                train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                t.set_postfix(loss=f"{loss:.4f}")

        train_loss /= len(train_data.dataset)
        t.set_postfix(loss=f"{train_loss:.4f}")
        train_loss_hist.append(train_loss)

        # Evaluate model on the validation set
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_data:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                val_loss += loss_func(pred, y).item()
        val_loss /= len(val_data.dataset)
        val_loss_hist.append(val_loss)
        print(f"[Epoch {epoch}] val_loss={val_loss:.6f}. ", end="")

        # Early stopping
        model.to("cpu")
        early_stopping(val_loss, model)
        model.to(device)
        if early_stopping.early_stop:
            print("Early stopping.")
            break

    return val_loss_hist, train_loss_hist
