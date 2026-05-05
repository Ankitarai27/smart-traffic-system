import numpy as np
import sounddevice as sd
from scipy.fft import fft

def detect_siren(duration=1, fs=44100):
    try:
        # Record audio
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()

        # Flatten audio
        audio = audio.flatten()

        # FFT (frequency analysis)
        freq_data = np.abs(fft(audio))

        # Get dominant frequency
        dominant_freq = np.argmax(freq_data)

        # 🚑 Siren frequency range (approx)
        if 1000 < dominant_freq < 3000:
            return True
        else:
            return False

    except:
        return False
    