import numpy as np


def generate_sine(freq, t, amp=1.0):
    # Generate a sine waveform of given frequency, time array t, and amplitude
    return amp * np.sin(2 * np.pi * freq * t)
