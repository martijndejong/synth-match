import numpy as np
from scipy.signal import butter, lfilter


def variable_lowpass_filter(data, cutoff_env, sample_rate, order=5, block_size=256, min_cutoff=1e-6):
    """
    Apply a variable low–pass filter using a block–based approach.

    This function assumes that the cutoff frequency remains nearly constant over
    a block of samples (of length block_size). For each block, it computes the
    average cutoff frequency from cutoff_env, designs a Butterworth filter using
    that average value (clamped to a minimum value), and applies lfilter to the block.
    The filter state is passed from one block to the next to maintain continuity.

    Parameters:
      data (np.ndarray): Input signal.
      cutoff_env (np.ndarray): Array of cutoff frequencies (Hz) per sample.
      sample_rate (float): Sample rate in Hz.
      order (int): Order of the Butterworth filter.
      block_size (int): Number of samples per block.
      min_cutoff (float): Minimum allowable cutoff frequency (Hz).

    Returns:
      np.ndarray: The filtered signal.
    """
    nyquist = 0.5 * sample_rate
    filtered_data = np.empty_like(data)
    zi = None  # filter state

    # Process data in blocks.
    for start in range(0, len(data), block_size):
        end = min(start + block_size, len(data))
        # Compute the average cutoff frequency over the current block.
        block_cutoff = np.mean(cutoff_env[start:end])
        # Clamp block_cutoff to a small positive value if it's too low.
        block_cutoff = max(block_cutoff, min_cutoff)
        normal_cutoff = block_cutoff / nyquist
        # Design the Butterworth filter with this (approximately constant) cutoff frequency.
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        # Initialize the filter state for the first block.
        if zi is None:
            zi = np.zeros(max(len(a), len(b)) - 1)
        # Filter the current block, updating the state.
        block_out, zi = lfilter(b, a, data[start:end], zi=zi)
        filtered_data[start:end] = block_out

    return filtered_data
