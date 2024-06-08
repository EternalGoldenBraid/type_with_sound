import numpy as np
import sounddevice as sd
from pynput import keyboard

blocksize = 1024*4

# Define the range of the keyboard characters
keyboard_chars = 'abcdefghijklmnopqrstuvwxyz'

# Define the frequency range (for example, from C4 to C6)
min_freq = 361.63  # C4
max_freq = 1046.5  # C6

# Calculate the frequency step
num_keys = len(keyboard_chars)
frequency_step = (max_freq - min_freq) / (num_keys - 1)

# Create the mapping from characters to frequencies
char_to_freq = {char: min_freq + i * frequency_step for i, char in enumerate(keyboard_chars)}
char_to_idx = {char: i for i, char in enumerate(keyboard_chars)}

sample_rate = 44100  # samples per second
duration = 50  # duration of the sine wave buffer in seconds
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
sine_waves = np.array([np.sin(2 * np.pi * frequency * t).astype(np.float32) for frequency in char_to_freq.values()])

current_volumes = np.zeros(num_keys)

current_key = 'c'

def callback(outdata, frames, time, status):
    global current_volumes
    # start_idx = int((time.currentTime * sample_rate) % len(sine_wave))
    # end_idx = (start_idx + frames) % len(sine_waves.shape[1])
    start_idx = int((time.currentTime * sample_rate) % sine_waves.shape[1])
    end_idx = (start_idx + frames) % sine_waves.shape[1]

    if start_idx < end_idx:
        outdata[:] = np.sum(
                        current_volumes.reshape(-1, 1) * sine_waves[:, start_idx:end_idx], axis=0
                        )[..., np.newaxis]
    else:  # handle wrap around
        outdata[:end_idx] = np.sum(
                                current_volumes.reshape(-1, 1) * sine_waves[:, :end_idx], axis=0
                                )[..., np.newaxis]

stream = sd.OutputStream(samplerate=sample_rate, channels=1, 
                         callback=callback, blocksize=blocksize)
stream.start()


def on_press(key):
    global current_volumes
    if hasattr(key, 'char') and key.char in char_to_idx.keys():
        current_volumes[char_to_idx[key.char]] = 0.2  # unmute on press
        print(current_volumes)
    # current_volume = 0.2  # unmute on press

def on_release(key):
    global current_volumes
    if hasattr(key, 'char') and key.char in char_to_idx.keys():
        current_volumes[char_to_idx[key.char]] = 0.0
    if key == keyboard.Key.esc:
        # Stop listener and exit
        return False

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

while True:
    pass
