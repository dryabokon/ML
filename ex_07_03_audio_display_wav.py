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
from librosa import tempo_frequencies
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
def edit_me():


    filename_in = 'D://au.mp3'
    #filename_out = 'D://part1.mp3'
    #X, fs = librosa.load(filename_in,offset=8,duration=19-8)

    #filename_out = 'D://part2.mp3'
    #X, fs = librosa.load(filename_in,offset=38,duration=52-38)

    #filename_out = 'D://part3.mp3'
    #X, fs = librosa.load(filename_in,offset=60+43,duration=51-43)

    #filename_out = 'D://part4.mp3'
    #X, fs = librosa.load(filename_in,offset=2*60+33,duration=45.5-33)

    #filename_out = 'D://part5.mp3'
    #X, fs = librosa.load(filename_in,offset=3*60+17,duration=25-17)

    #librosa.output.write_wav(filename_out, X, fs)

    #X, fs = librosa.load('D://part1.mp3')
    #librosa.output.write_wav('D://part1_trm.mp3', X[:-int(1.5*fs)], fs)

    res = []


    x1, fs = librosa.load('D://part1.mp3')
    x2, fs = librosa.load('D://part2.mp3')
    x3, fs = librosa.load('D://part3.mp3')
    x4, fs = librosa.load('D://part4.mp3')
    x5, fs = librosa.load('D://part5.mp3')

    res.append(numpy.zeros(fs*6,dtype=numpy.float32))
    res.append(x1[int(fs*0.5):-int(1.5*fs)])
    res.append(x2)
    res.append(numpy.zeros(fs * 5, dtype=numpy.float32))
    res.append(x3)
    res.append(x4)
    res.append(numpy.zeros(fs * 6, dtype=numpy.float32))
    res.append(x5)
    res = numpy.concatenate(res)

    bg, fs0 = librosa.load('D://bg_v2x.mp3')

    bg[:len(res)] = res*0.9 + bg[:len(res)]*0.1
    bg[len(res):]*=0.1

    librosa.output.write_wav('D://merged.mp3',bg, fs)
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #display_current_sound('./data/output/sound.wav')
    display_current_sound('./data/ex_wave/440.wav')
    #display_current_sound('./data/ex_wave/440_gen.wav')
    #display_current_sound('./data/ex_wave/clarinet_c6.wav')
    #display_current_sound('./data/ex_wave/prelude_cmaj.wav')
    #display_current_sound('./data/ex_wave/simple_piano.wav')

    #display_current_sound('./data/ex_wave/Sound_20140.mp3')
    #display_current_sound('./data/ex_wave/Sound_21728.mp3')

