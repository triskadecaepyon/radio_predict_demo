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


"""
Code to process spectrum and morse
from hamradio-toolkit
"""

def detect_signals(signal, threshold):
    """
    Detect signals and return the offset frequencies
    Uses a max signal threshold via np.max() to identify
    all the spectrum carrier freq. getting signals in the
    spectrogram
    Input: Post-fft Signal, and db threshold
    Output: List of detected signals by carrier freq, array mask
    """
    list_of_bool_offsets = signal.max(0) > threshold
    return list_of_bool_offsets


def process_carrier_data(signal, offset_mask):
    """
    
    """


def process_spectrum(signal, threshold):
    """
    Process Spectrum of Windowed Signal Data
    Input: Window spectrum data, Post-FFT
    Output: 
    """
    mask_list_of_offsets = detect_signals(signal, threshold)
    return mask_list_of_offsets

