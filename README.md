# radio_predict_demo
A demo showcasing the live analysis of radio signals with the SciPy stack and Machine Learning, with live controls to specify the size of the moving window utilized for ML training.  

Utilizes Bokeh and the SciPy stack for the majority of the demo.  In order to run the demo, use the command: bokeh serve --show spectrum_predict.py


## Minimum Requirements

- Bokeh > 0.12.13
- NumPy > 1.16.2
- SciPy > 1.2.1
- Python > 3.6

## How it works

This demo utilizes bokeh serve, which runs a bokeh server from the detected settings from the spectrum_predict.py file.  Please note that it does not run this file as main, which can cause some confusion when making edits or additions to this demo.

For the demo itself, it creates random RF noise on an RF carrier in the ham_audio.py functions, and then processes them via an FFT-both done via a periodic update on the bokeh server.  This data gets saved in an array which is a moving window/rolling index, which allows a history of FFT data to be analyzed.  This FFT data is sent to ham_audio.py's processing functions, which detect a minimum db value in each frequency slice of the FFT to create an array mask for detected signals.  This array mask is then sent to the machine learning classifer model as part of the training data for the chosen classifer, which in this case is a Support Vector Classifer (SVC).

On the rendering side, the base RF signal and the FFT signal are rendered in one row.  Above them is the computation time that the SVC model takes as well as the signal detection code.  The interactive slider changes the moving window size to create a larger training data set--with training time going up if the slider is moved up.  This example is essentially a streaming data model with a live retraining component running full-time.  
