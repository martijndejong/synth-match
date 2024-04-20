import numpy as np


def linear_interp(lower: float, upper: float, alpha: float = 0.5) -> float:
    """
    :param lower: lower value
    :param upper: upper value
    :param alpha: value between 0 and 1 (0 returns value_a, 1 returns value_b)
    :return: linear interpolated value at alpha
    """
    return lower + alpha * (upper - lower)


def mse(y_true, y_pred) -> float:
    """
    Calculate the mean squared error between y_true and y_pred
    :param y_true: true value
    :param y_pred: predicted value
    :return: mean squared error
    """
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred) -> float:
    """
    Calculate the root mean squared error between y_true and y_pred
    :param y_true: true value
    :param y_pred: predicted value
    :return: root mean squared error
    """
    return np.sqrt(mse(y_true, y_pred))
