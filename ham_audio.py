import numpy as np
import random
from scipy.io import wavfile
from scipy.fftpack import fft

"""
Simulated Signal generation
Code section partially from the example at
https://github.com/bokeh/bokeh/tree/master/examples/app/spectrogram
"""

NUM_SAMPLES = 1024
SAMPLING_RATE = 44100.
MAX_FREQ = SAMPLING_RATE / 2
FREQ_SAMPLES = NUM_SAMPLES / 8
TIMESLICE = 100  # ms
NUM_BINS = 16

_t = np.arange(0, NUM_SAMPLES/SAMPLING_RATE, 1.0/SAMPLING_RATE)
_f_carrier = 2000
_f_mod = 1000
_ind_mod = 1

def fm_modulation(x, f_carrier = 220, f_mod =220, Ind_mod = 1):
    y = np.sin(2*np.pi*f_carrier*x + Ind_mod*np.sin(2*np.pi*f_mod*x))
    return y


def make_audio():
    # Generate FM signal with drifting carrier and mod frequencies
    global _f_carrier, _f_mod, _ind_mod
    _f_carrier = max([_f_carrier+np.random.randn()*50, 0])
    _f_mod = max([_f_mod+np.random.randn()*20, 0])
    _ind_mod = max([_ind_mod+np.random.randn()*0.1, 0])
    A = 0.4 + 0.05 * np.random.random()
    signal = A * fm_modulation(_t, _f_carrier, _f_mod, _ind_mod)
    return signal


def create_signal_fft(signal):
    ffts = fft(signal)
    spectrum = abs(ffts)[:int(NUM_SAMPLES/2)]
    return spectrum
