import numpy as np
import random
from scipy.io import wavfile
from scipy.fftpack import fft
from bokeh.io import curdoc
from bokeh.layouts import row, column, gridplot
from bokeh.models import ColumnDataSource, Slider, Select
from bokeh.models.glyphs import Text
from bokeh.models.widgets import Slider, TextInput
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
    # Random Data for now

    # List of some real and some generated call signs
    ls_of_conv_callsigns = ['k5xrs','ki5ddl','k5gnu','k5pi','ae1x','n5nm','k5aes','k3nem']
    x_data = []
    y_data = []
    text_data = []
    for conversation_freq in range(0,50):
        x_data.append(conversation_freq)
        y_data.append(np.random.rand())
        text_data.append(random.choice(ls_of_conv_callsigns))

    return dict(x=x_data, y=y_data, text=text_data)

"""
Data Generation
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

    #status_bar_source.data = get_current_status()

    
def update_status():
    # A slower update component for less frequent data
    status_bar_source.data = get_current_status()


"""
Control Code, Sliders, and Interactivity
"""


window_size = Slider(title="Time Window Size (~1000 is 1 second) ", value=500, start=10.0, end=10000.0, step=1)
simul_signal_size = Slider(title="Simultaneous Signal allowance for ML", value=3, start=1, end=25, step=1)
random_morse_size = Slider(title="Number of Morse signals generated", value=3, start=1, end=25.0, step=1)

def update_data(attrname, old, new):

    print(window_size.value, simul_signal_size.value, random_morse_size.value)

for sliders in [window_size, simul_signal_size, random_morse_size]:
        sliders.on_change('value', update_data)


"""
Plotting Code
"""

# Code for the FFT plot
fft_plot = figure(tools="pan,wheel_zoom,box_zoom,reset,save",
                     y_axis_type="linear", y_range=[0,200], title="FFT of Audio Capture",
                     x_axis_label='Freq',x_range=[0,512], y_axis_label='Levels', plot_width=800, plot_height=300)

# Code for the signal plot
signal_plot = figure(tools="pan,wheel_zoom,box_zoom,reset,save",
                     y_axis_type="linear", y_range=[-0.8, 0.8], title="Signal diagram of Audio Capture",
                     x_axis_label='Freq',x_range=[0,512], y_axis_label='Levels', plot_width=500, plot_height=300)

# Code for the status plot
status_plot = figure(tools="pan,wheel_zoom,box_zoom,reset,save",
                     y_axis_type="linear", y_range=[0,1], title="Status",
                     x_axis_label='Freq',x_range=[0,50], y_axis_label='Detected Conversations', plot_width=1300, plot_height=150)

# Code for line and glyph generation

fft_plot.line(x='x', y='y', source=fft_source)
signal_plot.line(x='x', y='y', source=signal_source)

glyph = Text(x="x", y="y", text="text", text_color="#96deb3")
status_plot.add_glyph(status_bar_source, glyph)


"""
Final Layout and Callback generation
"""


audio_row = row(fft_plot, signal_plot, width=1500)
inputs = row(window_size, simul_signal_size, random_morse_size)
first_block = column(audio_row, status_plot, inputs)
layout = first_block

curdoc().add_root(layout)
curdoc().add_periodic_callback(update, 100)
curdoc().add_periodic_callback(update_status, 2000)
curdoc().title = "Radio Spectrum Prediction"
