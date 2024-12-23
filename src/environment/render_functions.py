import matplotlib.pyplot as plt
import numpy as np
# from src.environment.environment import Environment


def initialize_plots(rows, cols, figsize=(12, 8)):
    """Initialize a figure with a given number of subplots."""
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.ion()  # Turn on interactive mode
    plt.show()
    return fig, axes


def plot_spectrogram(ax, spectrogram, title, cmap='magma'):
    """Plot a spectrogram on a given axes."""
    ax.imshow(spectrogram, aspect='auto', origin='lower', cmap=cmap, interpolation='antialiased')
    ax.set_title(title)


def plot_spectrogram_multichannel(ax, spectrogram, title):
    """Plot a multichannel spectrogram on a given axes."""
    # Pad the third dimension with zeros to extend (x, y, 2) to (x, y, 3)
    spectrogram_rgb = np.pad(spectrogram, ((0, 0), (0, 0), (0, 1)), 'constant', constant_values=(0,))
    ax.imshow(spectrogram_rgb, aspect='auto', origin='lower', interpolation='antialiased')
    ax.set_title(title)


def plot_text(ax, param_names, current_params, target_params, reward, total_reward, episode, step):
    """Display text information on a given axes."""
    episode_info = f"Episode: {episode} | Total reward: {total_reward:.3f}"
    episode_info += f"\nCurrent Reward: {reward:.3f}"
    episode_info += f"\nCurrent step: {step}"

    synth_info = "Param name: current | target\n"
    synth_info += "\n".join(f"{param_name}: {current_params[index]:.3f} | {target_params[index]:.3f}"
                            for index, param_name in enumerate(param_names))

    ax.axis('off')
    ax.text(0.5, 0.9, episode_info, ha='center', va='center', fontsize=12, transform=ax.transAxes)
    ax.text(0.5, 0.75, synth_info, ha='center', va='top', fontsize=8, transform=ax.transAxes)


def update_plots(env):
    """
    Main render function called by environment.
    This function calls the other specific plotting functions
    Update plots with new data.
    """
    axes = env.axes.flatten()  # Flatten in case it's a matrix of axes
    for ax in axes:
        ax.clear()
    plot_spectrogram(
        ax=axes[0],
        spectrogram=env.target_audio.spectrogram,
        title="Target Audio Spectrogram"
    )
    plot_text(
        ax=axes[1],
        param_names=env.param_names,
        current_params=env.current_params,
        target_params=env.target_params,
        reward=env.last_reward,
        total_reward=env.total_reward,
        episode=env.episode,
        step=env.step_count
    )
    plot_spectrogram(
        ax=axes[2],
        spectrogram=env.current_audio.spectrogram,
        title="Current Synth Audio Spectrogram"
    )
    plot_spectrogram_multichannel(
        ax=axes[3],
        spectrogram=np.stack((env.current_audio.spectrogram, env.target_audio.spectrogram), axis=-1),
        title="Observed State Spectrogram"
    )
    plt.draw()
    plt.pause(0.001)
