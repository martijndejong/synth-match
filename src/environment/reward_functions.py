# Functions for calculating the reward based on audio similarity
import numpy as np
from skimage.metrics import structural_similarity as ssim

from src.utils.math import mse, rmse


def calculate_ssim(image1, image2):
    """
    Calculate the Structural Similarity Index (SSIM) between two single-channel images using scikit-image.

    Parameters:
    image1 (numpy.ndarray): First image (grayscale, numpy array).
    image2 (numpy.ndarray): Second image (grayscale, numpy array).

    Returns:
    float: SSIM index between the two images.
    """
    # Ensure the images are in float format
    image1 = image1.astype('float32')
    image2 = image2.astype('float32')

    # Calculate SSIM
    ssim_value = ssim(image1, image2)

    return ssim_value


def masked_ssim(target_spectrogram, current_spectrogram, threshold=0.1):
    """
    Calculate a modified Structural Similarity Index (SSIM) that focuses on the 'active' parts of the spectrogram,
    ignoring less significant areas.

    Parameters:
    target_spectrogram (numpy.ndarray): The target spectrogram (2D array).
    current_spectrogram (numpy.ndarray): The current spectrogram to compare against the target (2D array).
    threshold (float): Threshold value to determine active areas in the target spectrogram.

    Returns:
    float: Modified SSIM value focusing on active areas.
    """
    # Convert spectrograms to float
    target_spectrogram = target_spectrogram.astype(np.float32)
    current_spectrogram = current_spectrogram.astype(np.float32)

    # Generate a mask from the spectrograms
    mask = (current_spectrogram >= threshold) | (target_spectrogram >= threshold)

    # Apply mask to both spectrograms
    masked_current = current_spectrogram[mask]
    masked_target = target_spectrogram[mask]

    # Calculate SSIM on masked areas
    ssim_value = ssim(masked_target, masked_current, data_range=masked_target.max() - masked_target.min())

    return ssim_value


def weighted_ssim(target_spectrogram, current_spectrogram):
    """
    Calculate a weighted Structural Similarity Index (SSIM) that emphasizes differences in higher energy areas.

    Parameters:
    target_spectrogram (numpy.ndarray): The target spectrogram (2D array, float).
    current_spectrogram (numpy.ndarray): The current spectrogram to compare against the target (2D array, float).

    Returns:
    float: Weighted SSIM value focusing on areas with higher energy.
    """
    # Ensure the spectrograms are in float format and normalize if not already
    target_spectrogram = target_spectrogram.astype(np.float32)
    current_spectrogram = current_spectrogram.astype(np.float32)

    # Generate weights from the target spectrogram
    weights = np.where(target_spectrogram > target_spectrogram.mean(), target_spectrogram, 0)
    weights /= weights.max()  # Normalize weights to range [0, 1]

    # Calculate weighted SSIM
    ssim_value = ssim(target_spectrogram, current_spectrogram,
                      data_range=target_spectrogram.max() - target_spectrogram.min(), sample_weight=weights)

    return ssim_value


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


def mse_similarity_masked(current_sample, target_sample, threshold=0.1):
    """
    Calculate the similarity between two np arrays using Mean Squared Error (MSE).
    """
    mask = (current_sample >= threshold) | (target_sample >= threshold)
    masked_current = current_sample[mask]
    masked_target = target_sample[mask]

    mse_value = mse(y_true=masked_target, y_pred=masked_current)

    # Convert MSE to a similarity measure; lower MSE should yield higher similarity
    similarity = np.exp(-mse_value)  # Using exponential to ensure a positive similarity score
    return similarity


def rmse_similarity_masked(current_sample, target_sample, threshold=0.1):
    mask = (current_sample >= threshold) | (target_sample >= threshold)
    masked_current = current_sample[mask]
    masked_target = target_sample[mask]

    # Ensure there are enough values left after masking
    if len(masked_current) == 0 or len(masked_target) == 0:
        return 0

    return rmse_similarity(masked_current, masked_target)


def time_cost(step_count, factor=0.1):
    return factor * step_count


def action_cost(action: np.ndarray, factor: float = 10.0):
    return factor * np.sum((action) ** 2)


def saturation_penalty(synth_params, actions, penalty_amount=10.0):
    """
    Calculate a penalty based on the actions that would saturate the synth parameters.

    Args:
    - synth_params (np.ndarray): Current synth parameters (shape: (num_params,)).
    - actions (np.ndarray): Actions to be taken (shape: (num_params,)).
    - penalty_amount (float): The penalty amount for saturating a parameter.

    Returns:
    - penalty (float): Total penalty for the given actions.
    """
    # Calculate the resulting parameters after applying the actions
    new_params = synth_params + actions

    # Calculate the amount of oversaturation for each parameter
    overshoot_below = np.minimum(0, new_params)  # Amount below 0
    overshoot_above = np.maximum(1, new_params) - 1  # Amount above 1

    # Calculate the total oversaturation
    total_overshoot = np.abs(overshoot_below) + np.abs(overshoot_above)

    # Calculate the penalty
    penalty = np.sum(total_overshoot) * penalty_amount

    return penalty
