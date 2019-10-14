import numpy as np
from scipy.io import wavfile
from scipy.fftpack import fft
from bokeh.io import curdoc
from bokeh.layouts import row, column, gridplot
from bokeh.models import ColumnDataSource, Slider, Select
from bokeh.plotting import curdoc, figure


"""
Simulated Signal generation
Code section partially from the example at
https://github.com/bokeh/bokeh/tree/master/examples/app/spectrogram
"""


def fm_modulation(x, f_carrier = 220, f_mod =220, Ind_mod = 1):
    y = np.sin(2*np.pi*f_carrier*x + Ind_mod*np.sin(2*np.pi*f_mod*x))
    return y

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


def get_current_status():
    return None


"""
Plot Generation
"""

signal_source = ColumnDataSource(data=dict(x=[], y=[]))
fft_source = ColumnDataSource(data=dict(x=[], y=[]))
status_bar_source = ColumnDataSource(data=dict(x=[], y=[], text=[]))

def update():
    signal_data = make_audio()
    fft_data = create_signal_fft(signal_data)
    x_data = [x for x in range(0,512)]
    y_data = fft_data.reshape(512,1).tolist()

    fft_source.data = dict(x=x_data, y=y_data)
    signal_y_data = signal_data[::2].reshape(512,1).tolist()

    signal_source.data = dict(x=x_data, y=signal_y_data)

    status_bar_source = get_current_status()


# Code for the FFT plot
fft_plot = figure(tools="pan,wheel_zoom,box_zoom,reset,save",
                     y_axis_type="linear", y_range=[0,200], title="FFT of Audio Capture",
                     x_axis_label='Freq',x_range=[0,512], y_axis_label='Levels', plot_width=800, plot_height=400)

# Code for the signal plot
signal_plot = figure(tools="pan,wheel_zoom,box_zoom,reset,save",
                     y_axis_type="linear", y_range=[-0.8, 0.8], title="Signal diagram of Audio Capture",
                     x_axis_label='Freq',x_range=[0,512], y_axis_label='Levels', plot_width=500, plot_height=400)

status_plot = figure(tools="pan,wheel_zoom,box_zoom,reset,save",
                     y_axis_type="linear", y_range=[0,200], title="Status",
                     x_axis_label='Freq',x_range=[0,512], y_axis_label='Levels', plot_width=500, plot_height=400)

fft_plot.line(x='x', y='y', source=fft_source)
signal_plot.line(x='x', y='y', source=signal_source)
status_plot.line(x='x', y='y', source=signal_source)

audio_row = row(fft_plot, signal_plot, width=5000)

layout = column(audio_row, status_plot)

curdoc().add_root(layout)
curdoc().add_periodic_callback(update, 100)
curdoc().title = "Radio Spectrum Prediction"
