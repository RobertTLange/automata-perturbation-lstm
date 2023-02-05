import numpy as np
from torch.utils import data


def get_train_test_split_ids(
    num_data_points: int, test_set_size: float
) -> np.ndarray:
    """Generate single train test split ids for datapoints."""
    test_points = int(np.round(num_data_points * test_set_size))
    train_points = num_data_points - test_points
    train_test_ids = np.array(train_points * [1] + test_points * [0])
    np.random.shuffle(train_test_ids)
    return train_test_ids


def get_train_test_split_loaders(
    X: np.ndarray, y: np.ndarray, train_test_split: float, batch_size: int
):
    """Split the np arrays and return torch dataloaders."""
    split = get_train_test_split_ids(X.shape[0], train_test_split)
    train_ids = np.argwhere(split == 1).squeeze()
    test_ids = np.argwhere(split == 0).squeeze()
    train_loader, test_loader = wrap_data_torch(
        X, y, train_ids, test_ids, batch_size
    )
    return train_loader, test_loader


def wrap_data_torch(
    X: np.ndarray,
    y: np.ndarray,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    batch_size: int,
):
    X_train, y_train = X[train_ids], y[train_ids]
    X_test, y_test = X[test_ids], y[test_ids]

    # Wrap data in torch generator & dataloader objects
    training_set = Dataset(X_train, y_train)
    test_set = Dataset(X_test, y_test)

    # Set parameters for the dataloaders
    train_params = {"batch_size": batch_size, "shuffle": True, "num_workers": 3}
    test_params = {"batch_size": 100, "shuffle": True, "num_workers": 3}

    train_loader = data.DataLoader(training_set, **train_params)
    test_loader = data.DataLoader(test_set, **test_params)
    return train_loader, test_loader


class Dataset(data.Dataset):
    """Simple Dataset Wrapper for your Data"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """Wrap the data in the dataset torch wrapper"""
        self.X = X
        self.y = y

    def __len__(self):
        """Get the number of samples in the buffer"""
        return self.X.shape[0]

    def __getitem__(self, index):
        """Get one sample from the dataset"""
        X = self.X[index, ...]
        y = self.y[index, ...]
        return X, y
