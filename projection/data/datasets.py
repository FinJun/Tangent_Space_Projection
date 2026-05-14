"""Dataset utilities."""

import pyepo


def build_datasets(x, c, opt_model, n_train, n_val, n_test):
    """Split data and build PyEPO datasets."""
    x_train, c_train = x[:n_train], c[:n_train]
    x_val, c_val = x[n_train:n_train+n_val], c[n_train:n_train+n_val]
    x_test, c_test = x[-n_test:], c[-n_test:]

    trainset = pyepo.data.dataset.optDataset(opt_model, x_train, c_train)
    valset = pyepo.data.dataset.optDataset(opt_model, x_val, c_val)
    testset = pyepo.data.dataset.optDataset(opt_model, x_test, c_test)

    return trainset, valset, testset
