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


def calculate_ssim_with_mask(target_spectrogram, current_spectrogram, threshold=0.1):
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

    # Generate a mask from the target spectrogram
    mask = target_spectrogram > threshold

    # Apply mask to both spectrograms
    masked_target = np.where(mask, target_spectrogram, 0)
    masked_current = np.where(mask, current_spectrogram, 0)

    # Calculate SSIM on masked areas
    ssim_value = ssim(masked_target, masked_current, data_range=masked_target.max() - masked_target.min())

    return ssim_value


def calculate_weighted_ssim(target_spectrogram, current_spectrogram):
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


def time_cost(step_count, factor=0.1):
    return factor * step_count


def action_cost(action: np.ndarray, factor: float = 10.0):
    return factor * np.sum((action) ** 2)
