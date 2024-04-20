# Functions for calculating the reward based on audio similarity
import numpy as np
from src.utils.math import mse, rmse


def mse_similarity(current_sample, target_sample):
    """
    Calculate the similarity between two np arrays using Mean Squared Error (MSE).
    """
    mse_value = mse(y_true=target_sample, y_pred=current_sample)

    # Convert MSE to a similarity measure; lower MSE should yield higher similarity
    similarity = np.exp(-mse_value)  # Using exponential to ensure a positive similarity score
    return similarity


def rmse_similarity(current_sample, target_sample):
    """
    Calculate the similarity between two np arrays using Root Mean Squared Error (RMSE).
    """
    rmse_value = rmse(y_true=target_sample, y_pred=current_sample)

    # Convert MSE to a similarity measure; lower MSE should yield higher similarity
    similarity = np.exp(-rmse_value)  # Using exponential to ensure a positive similarity score
    return similarity


def time_cost(step_count, factor=0.1):
    return factor * step_count


def action_cost(action: np.ndarray, factor: float = 10.0):
    return factor * np.sum((action) ** 2)
