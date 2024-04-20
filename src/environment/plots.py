import matplotlib.pyplot as plt


def initialize_plots(rows, cols, figsize=(12, 8)):
    """Initialize a figure with a given number of subplots."""
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.ion()  # Turn on interactive mode
    plt.show()
    return fig, axes


def plot_spectrogram(ax, spectrogram, title, cmap='magma'):
    """Plot a spectrogram on a given axes."""
    ax.imshow(spectrogram, aspect='auto', origin='lower', cmap=cmap)
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


def update_plots(axes, target_spectrogram, current_spectrogram, state_spectrogram, param_names,
                 current_params, target_params, reward, total_reward, episode, step):
    """Update plots with new data."""
    axes = axes.flatten()  # Flatten in case it's a matrix of axes
    for ax in axes:
        ax.clear()
    plot_spectrogram(axes[0], target_spectrogram, "Target Audio Spectrogram")
    plot_text(axes[1], param_names, current_params, target_params, reward, total_reward, episode, step)
    plot_spectrogram(axes[2], current_spectrogram, "Current Synth Audio Spectrogram")
    plot_spectrogram(axes[3], state_spectrogram, "Observed State Spectrogram")
    plt.draw()
    plt.pause(0.1)
