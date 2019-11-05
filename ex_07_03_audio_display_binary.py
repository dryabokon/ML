# ----------------------------------------------------------------------------------------------------------------------
import cv2
import numpy
import pyaudio
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal
matplotlib.rcParams['toolbar'] = 'None'
# ----------------------------------------------------------------------------------------------------------------------
form_1 = pyaudio.paInt16
chans = 1
samp_rate = 44100
chunk = 4096
fs = 44110
# ----------------------------------------------------------------------------------------------------------------------
g_key=' '
# ----------------------------------------------------------------------------------------------------------------------
def press(event):
    global g_key
    g_key = event.key
    return
# ----------------------------------------------------------------------------------------------------------------------
def display_current_sound0(filename_in):

    plt.ion()
    fig = plt.figure()
    fig.canvas.mpl_connect('key_press_event', press)

    subplot1 = fig.add_subplot(131)
    subplot2 = fig.add_subplot(132)
    subplot3 = fig.add_subplot(133)
    subplot1_data, subplot2_data, subplot3_data = None, None, None

    subplot1.tick_params(axis='both', left='off', top='off', right='off', bottom='on', labelleft='off', labeltop='off',labelright='off', labelbottom='on')
    subplot2.tick_params(axis='both', left='on', top='off', right='off', bottom='on', labelleft='on', labeltop='off',labelright='off', labelbottom='on')
    subplot1.grid(which='major', color='lightgray', linestyle='--')
    subplot2.grid(which='major', color='lightgray', linestyle='--')
    subplot1.set_ylim([-0x7FFF, 0x7FFF])
    subplot2.set_xlim([0, 1024])
    subplot2.set_ylim([0, 2000])
    #subplot3.set_ylim([0, 2000])

    subplot1.set_title('Signal 1')
    subplot2.set_title('Frequence responce')
    subplot3.set_title('Spectrogram')

    plt.tight_layout()


    N=10
    i=0
    while True:
        f_handle = open(filename_in, "rb");data = f_handle.read();f_handle.close()
        X = numpy.fromstring(data, dtype=numpy.int16)
        size = len(X)

        if subplot1_data is None:
            subplot1_data, = subplot1.plot(X)
            X_hist = numpy.zeros((len(X)*N))

        if subplot2_data is None:
            f0, Pxx_spec = signal.welch(X, fs, 'flattop', chunk, scaling='spectrum')
            Pxx_spec = numpy.sqrt(Pxx_spec)
            subplot2_data, = subplot2.plot(Pxx_spec, f0)

        if i%N ==0:
            subplot3.specgram(X_hist, NFFT=chunk, Fs=fs,vmin=0, vmax=50)
            subplot3.set_ylim([0, 2000])
            plt.tight_layout()
            i=0


        if len(X)!=size:continue

        subplot1_data.set_ydata(X)
        f, Pxx_spec = signal.welch(X, fs, 'flattop', chunk, scaling='spectrum')
        Pxx_spec = numpy.sqrt(Pxx_spec)
        if len(f)!=len(f0):continue

        subplot2_data.set_xdata(Pxx_spec)
        subplot2_data.set_ydata(f)

        X_hist[i*len(X):(i+1)*len(X)]=X

        subplot1.set_title('%d'%i)
        fig.canvas.draw()
        fig.canvas.flush_events()
        i += 1
        if g_key == 'escape':break

    return
# ----------------------------------------------------------------------------------------------------------------------
def signal_to_image(X,image,low,high):
    image[:,:,:]=255
    for i,x in enumerate(X):
        row = int((x-low)*image.shape[0]/(high-low))
        col = int(i*image.shape[1]/len(X))
        image[row,col,:]=(255,128,0)
    return image
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    display_current_sound0('./data/output/sound.dat')



