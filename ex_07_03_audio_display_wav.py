# ----------------------------------------------------------------------------------------------------------------------
import numpy
import pyaudio
import scipy
import matplotlib
import matplotlib.pyplot as plt
#import wave
from scipy.io import wavfile
from playsound import playsound
from scipy import signal
import librosa
import tools_audio
# ----------------------------------------------------------------------------------------------------------------------
matplotlib.rcParams['toolbar'] = 'None'
# ----------------------------------------------------------------------------------------------------------------------
form_1 = pyaudio.paInt16
chans = 1
samp_rate = 44100
chunk = 4096
# ----------------------------------------------------------------------------------------------------------------------
g_key=' '
# ----------------------------------------------------------------------------------------------------------------------
def press(event):
    global g_key
    g_key = event.key
    return
# ----------------------------------------------------------------------------------------------------------------------
def swap(line2d,xdata, ydata):
    line2d.set_xdata(ydata)
    line2d.set_ydata(xdata)
    return
# ----------------------------------------------------------------------------------------------------------------------
def display_current_sound(filename_in):

    plt.ion()
    fig = plt.figure()
    fig.canvas.mpl_connect('key_press_event', press)

    subplot1 = fig.add_subplot(131)
    subplot2 = fig.add_subplot(132)
    subplot3 = fig.add_subplot(133)
    subplot1_data, subplot2_data,subplot3_data = None, None, None

    subplot1.tick_params(axis='both', left='off', top='off', right='off', bottom='on', labelleft='off', labeltop='off',labelright='off', labelbottom='on')
    subplot2.tick_params(axis='both', left='on', top='off', right='off', bottom='on', labelleft='on', labeltop='off',labelright='off', labelbottom='on')
    subplot1.grid(which='major', color='lightgray', linestyle='--')
    subplot2.grid(which='major', color='lightgray', linestyle='--')
    subplot1.set_ylim([-0x7FFF, 0x7FFF])
    subplot2.set_xlim([0, 1024])
    subplot2.set_ylim([0, 2000])
    subplot3.set_ylim([0, 2000])

    subplot1.set_title('Signal 1')
    subplot2.set_title('Frequence responce')
    subplot3.set_title('Spectrogram')

    plt.tight_layout()

    while True:
        #fs0, Y = wavfile.read(filename_in)
        X, fs = librosa.load(filename_in)

        X*= 0x7FFF
        playsound(filename_in, False)
        start, stop = 0, chunk

        if subplot1_data is None:
            subplot1_data, = subplot1.plot(numpy.array(X[start:stop], dtype=numpy.float))

        if subplot2_data is None:
            f, Pxx_spec = signal.welch(X[start:stop], fs, 'flattop', chunk, scaling='spectrum')
            subplot2_data, = subplot2.plot(numpy.sqrt(Pxx_spec),f)

        if subplot3_data is None:
            subplot3_data = subplot3.specgram(X, NFFT=chunk, Fs=fs)
            subplot3.set_ylim([0, 2000])

        fig.canvas.draw()
        fig.canvas.flush_events()

        while stop<len(X):
            start+=chunk//2
            stop+=chunk//2

            data_array = numpy.array(X[start:stop],dtype=numpy.float)

            if len(data_array)!=chunk:continue
            subplot1_data.set_ydata(data_array)

            f, Pxx_spec = signal.welch(X[start:stop], fs, 'flattop', chunk, scaling='spectrum')
            subplot2_data.set_xdata(numpy.sqrt(Pxx_spec))
            subplot2_data.set_ydata(f)

            fig.canvas.draw()
            fig.canvas.flush_events()
            if g_key == 'escape': return

        if g_key == 'escape': return

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #display_current_sound('./data/output/sound.wav')
    #display_current_sound('./data/ex_wave/440.wav')
    #display_current_sound('./data/ex_wave/440_gen.wav')
    #display_current_sound('./data/ex_wave/clarinet_c6.wav')
    #display_current_sound('./data/ex_wave/prelude_cmaj.wav')
    #display_current_sound('./data/ex_wave/simple_piano.wav')

    display_current_sound('./data/ex_wave/Sound_20140.mp3')
    #display_current_sound('./data/ex_wave/Sound_21728.mp3')





