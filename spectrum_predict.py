import numpy as np
import random

from bokeh.io import curdoc
from bokeh.layouts import row, column, gridplot
from bokeh.models import ColumnDataSource, Slider, Select
from bokeh.models.glyphs import Text
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import curdoc, figure
from bokeh.driving import count

import ml_model
from ham_audio import make_audio, create_signal_fft, fm_modulation, process_spectrum


"""
General functions
"""


def mv_window_add(np_array, data, index):
    np_array[index] = data


def mv_window_view(np_array, index):
    return np.roll(np_array, index)


def get_current_status():
    # Random Data for now

    # List of some real and some generated call signs
    ls_of_conv_callsigns = ['k5xrs','ki5ddl','k5gnu','k5pi','ae1x',
                            'n5nm','k5aes','k3nem']
    x_data = []
    y_data = []
    text_data = []
    for conversation_freq in range(0,50):
        x_data.append(conversation_freq)
        y_data.append(np.random.rand())
        text_data.append(random.choice(ls_of_conv_callsigns))

    return dict(x=x_data, y=y_data, text=text_data)

@count()
def update(t):
    signal_data = make_audio()
    fft_data = create_signal_fft(signal_data)
    x_data = [x for x in range(0,512)]
    y_data = fft_data.reshape(512,1).tolist()

    fft_source.data = dict(x=x_data, y=y_data)
    signal_y_data = signal_data[::2].reshape(512,1).tolist()

    mv_window_add(signal_window_data,fft_data,t % 500)
    ref_index = t % 500 # Use the modulo for a wrapping index
    window_source.data = dict(index=[ref_index])
    signal_source.data = dict(x=x_data, y=signal_y_data)


def update_status():
    # A slower update component for less frequent data
    status_bar_source.data = get_current_status()


def update_data(attrname, old, new):

    # Placeholder for now
    print(window_size.value, simul_signal_size.value, random_morse_size.value)


def update_ml():

    print("running ML")
    if signal_window_data.shape[0] != window_size.value:
        print("different size")
        signal_window_data.resize(window_size.value,512)

    #print(signal_window_data.shape)
    process_spectrum(signal_window_data, 10)
    for_ml = mv_window_view(signal_window_data, window_source.data['index'])


"""
Bokeh Streaming Sources
"""

signal_source = ColumnDataSource(data=dict(x=[], y=[]))
fft_source = ColumnDataSource(data=dict(x=[], y=[]))
status_bar_source = ColumnDataSource(data=dict(x=[], y=[], text=[]))
window_source = ColumnDataSource(data=dict(index=[]))
signal_window_data = np.zeros([500,512])
ref_index = 0

#signal_window_data = np.zeros([1000,100])

"""
Bokeh Plotting Code
"""

# Code for the FFT plot
fft_plot = figure(tools="pan,wheel_zoom,box_zoom,reset,save",
                  y_axis_type="linear", y_range=[0,200],
                  title="FFT of Audio Capture", x_axis_label='Freq',
                  x_range=[0,512], y_axis_label='Levels',
                  plot_width=800, plot_height=300)

# Code for the signal plot
signal_plot = figure(tools="pan,wheel_zoom,box_zoom,reset,save",
                     y_axis_type="linear", y_range=[-0.8, 0.8],
                     title="Signal diagram of Audio Capture",
                     x_axis_label='Freq', x_range=[0,512], y_axis_label='Levels',
                     plot_width=500, plot_height=300)

# Code for the status plot
status_plot = figure(tools="pan,wheel_zoom,box_zoom,reset,save",
                     y_axis_type="linear", y_range=[0,1], title="Status",
                     x_axis_label='Freq', x_range=[0,50],
                     y_axis_label='Detected Conversations',
                     plot_width=1300, plot_height=150)


"""
Control Code, Sliders, and Interactivity
"""


window_size = Slider(title="Time Window Size (~1000 is 1 second) ", value=500,
                     start=10.0, end=10000.0, step=1)
simul_signal_size = Slider(title="Simultaneous Signal allowance for ML",
                           value=3, start=1, end=25, step=1)
random_morse_size = Slider(title="Number of Morse signals generated", value=3,
                           start=1, end=25.0, step=1)


"""
Code for line and glyph generation
"""

fft_plot.line(x='x', y='y', source=fft_source)
signal_plot.line(x='x', y='y', source=signal_source)

glyph = Text(x="x", y="y", text="text", text_color="#96deb3")
status_plot.add_glyph(status_bar_source, glyph)


"""
Final Layout and Callback generation
"""


# Register sliders for callbacks
for sliders in [window_size, simul_signal_size, random_morse_size]:
    sliders.on_change('value', update_data)

audio_row = row(fft_plot, signal_plot, width=1500)
inputs = row(window_size, simul_signal_size, random_morse_size)
first_block = column(audio_row, status_plot, inputs)
layout = first_block

curdoc().add_root(layout)
curdoc().add_periodic_callback(update, 100)
curdoc().add_periodic_callback(update_status, 2000)
curdoc().add_periodic_callback(update_ml, 2000)
curdoc().title = "Radio Spectrum Prediction"
