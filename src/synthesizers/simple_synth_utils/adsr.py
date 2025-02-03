import numpy as np


def generate_adsr(attack, decay, sustain, release, duration, sample_rate):
    """
    Generate an ADSR envelope whose values range from 0 to 1.

    The envelope is constructed as follows:
      - Attack: ramp from 0 to 1 over the attack time.
      - Decay: ramp from 1 down to the sustain level over the decay time.
      - Sustain: hold the sustain level for the remainder of the note–on period.
      - Release: ramp from the sustain level to 0 over the release time.

    If the note is released before the attack and/or decay phases complete, then
    only the available portion is generated before beginning the release.

    Parameters:
      attack (float): Attack time in seconds.
      decay (float): Decay time in seconds.
      sustain (float): Sustain level (0.0 to 1.0).
      release (float): Release time in seconds.
      duration (float): Duration of the note “on” (in seconds, not including release).
      sample_rate (float): Sample rate in Hz.

    Returns:
      envelope (np.ndarray): An array of envelope values (range 0–1). Its length is
         note_on_samples + release_samples.
    """
    attack_samples = int(attack * sample_rate)
    decay_samples = int(decay * sample_rate)
    release_samples = int(release * sample_rate)
    note_on_samples = int(duration * sample_rate)

    # Case 1: Note ends during the attack phase.
    if note_on_samples <= attack_samples:
        attack_env = np.linspace(0, 1, attack_samples, endpoint=False)[:note_on_samples]
        current = attack_env[-1] if len(attack_env) > 0 else 1.0
        release_env = np.linspace(current, 0, release_samples, endpoint=False)
        envelope = np.concatenate([attack_env, release_env])
    # Case 2: Note ends during the decay phase.
    elif note_on_samples <= (attack_samples + decay_samples):
        attack_env = np.linspace(0, 1, attack_samples, endpoint=False)
        remaining = note_on_samples - attack_samples
        full_decay = np.linspace(1, sustain, decay_samples, endpoint=False)
        decay_env = full_decay[:remaining]
        envelope = np.concatenate([attack_env, decay_env])
        current = envelope[-1] if len(envelope) > 0 else sustain
        release_env = np.linspace(current, 0, release_samples, endpoint=False)
        envelope = np.concatenate([envelope, release_env])
    # Case 3: Full attack, decay, and sustain phases.
    else:
        attack_env = np.linspace(0, 1, attack_samples, endpoint=False)
        decay_env = np.linspace(1, sustain, decay_samples, endpoint=False)
        sustain_samples = note_on_samples - (attack_samples + decay_samples)
        sustain_env = np.full(sustain_samples, sustain)
        on_env = np.concatenate([attack_env, decay_env, sustain_env])
        release_env = np.linspace(sustain, 0, release_samples, endpoint=False)
        envelope = np.concatenate([on_env, release_env])

    return envelope
