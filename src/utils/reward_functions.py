# Functions for calculating the reward based on audio similarity
import numpy as np


def calculate_mse_similarity(current_sample, target_sample):
    """
    Calculate the similarity between two np arrays using Mean Squared Error (MSE).
    """
    # TODO: should MSE somehow be normalized to number of data points?
    mse = np.mean((target_sample - current_sample) ** 2)  # / current_sample.size

    # Convert MSE to a similarity measure; lower MSE should yield higher similarity
    similarity = np.exp(-mse)  # Using exponential to ensure a positive similarity score
    return similarity


def time_penalty(step_count, beta=0.1):
    return beta * step_count

