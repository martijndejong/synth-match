import numpy as np


def variable_lowpass_filter(data, cutoff_env, sample_rate):
    """
    Apply a simple one–pole low–pass filter whose cutoff frequency varies over time.

    The filter uses the following recurrence for each sample n:
       y[n] = alpha[n] * x[n] + (1 - alpha[n]) * y[n-1],
    where:
       alpha[n] = dt / (RC + dt)   with   RC = 1 / (2*pi*cutoff[n])   and   dt = 1/sample_rate

    Parameters:
      data (np.ndarray): Input signal.
      cutoff_env (np.ndarray): Array of cutoff frequencies (Hz) for each sample.
      sample_rate (float): Sample rate in Hz.

    Returns:
      filtered (np.ndarray): The filtered signal.
    """
    dt = 1.0 / sample_rate
    filtered = np.zeros_like(data)
    filtered[0] = data[0]
    for n in range(1, len(data)):
        fc = cutoff_env[n]
        # Prevent division by zero or extremely low cutoff values:
        if fc < 1e-6:
            fc = 1e-6
        RC = 1.0 / (2 * np.pi * fc)
        alpha = dt / (RC + dt)
        filtered[n] = alpha * data[n] + (1 - alpha) * filtered[n - 1]
    return filtered
