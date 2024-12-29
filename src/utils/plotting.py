import numpy as np
import matplotlib.pyplot as plt

# Set up audio plotting scripts
def compare_time_plots(sig1, sig2, sampling_freq, figsize=(12, 6)):

    # Ensure the two signals have the same length
    if len(sig1) != len(sig2):
        raise ValueError(f"Length of the two signals does not match: {len(sig1)} vs {len(sig2)}")
    
    # Create time vector
    time = np.linspace(0, len(sig1) / sampling_freq, num=len(sig1))
    
    # Plot both signals in a single plot
    plt.figure(figsize=figsize)
    plt.plot(time, sig1, label="Target Audio", color="blue")
    plt.plot(time, sig2, label="Current Audio", color="orange")
    
    # Add plot details
    plt.title("Comparison of Target and Current Audio")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    
    # Show the plot
    plt.tight_layout()
    plt.show()

def compare_frequency_spectrograms(target_spectrogram, current_spectrogram, figsize=(12, 8)):
    # Create the figure and axes
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    
    # Plot target spectrogram
    im1 = axes[0].imshow(
        target_spectrogram,
        aspect='auto',
        origin='lower',
        cmap='magma',
        interpolation='antialiased'
    )
    axes[0].set_title('Target Spectrogram')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Frequency')
    
    # Add colorbar for the first spectrogram
    cbar1 = fig.colorbar(im1, ax=axes[0], orientation='vertical', shrink=0.8, pad=0.02)
    cbar1.set_label('Amplitude')

    # Plot current spectrogram
    im2 = axes[1].imshow(
        current_spectrogram,
        aspect='auto',
        origin='lower',
        cmap='magma',
        interpolation='antialiased'
    )
    axes[1].set_title('Current Spectrogram')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Frequency')

    # Add colorbar for the second spectrogram
    cbar2 = fig.colorbar(im2, ax=axes[1], orientation='vertical', shrink=0.8, pad=0.02)
    cbar2.set_label('Amplitude')

    # Show the plot
    plt.show()