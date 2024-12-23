class Host:
    def __init__(self, synthesizer, sample_rate=44100.):
        """
        Initialize the host with a given synthesizer object.

        Parameters:
            synthesizer (class): A synthesizer class (e.g., SuperSimpleSynth).
            sample_rate (float): The sample rate for the synthesizer (default: 44100 Hz).
        """
        self.vst = synthesizer(sample_rate=sample_rate)

    def play_note(self, note, note_duration):
        """
        Play a note using the loaded synthesizer.

        Parameters:
            note (str or int): The note to play (e.g., 'C4' or MIDI number).
            note_duration (float): Duration of the note in seconds.

        Returns:
            np.ndarray: The generated sound data.
        """
        return self.vst.play_note(note=note, duration=note_duration)


# Example Usage
if __name__ == "__main__":
    from super_simple_synth import SuperSimpleSynth

    # Initialize host with SuperSimpleSynth
    host = Host(synthesizer=SuperSimpleSynth, sample_rate=44100)

    # Play a note
    sound_data = host.play_note('C4', 2.0)
    print("Generated sound data:", sound_data)
