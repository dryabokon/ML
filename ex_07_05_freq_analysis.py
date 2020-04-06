import cv2
from matplotlib import pyplot as plt
import numpy
import tools_signal
import tools_IO
# ----------------------------------------------------------------------------------------------------------------------
def generate_time_vec(duration,time_step):
    time_vec = numpy.arange(0, duration, time_step)
    return time_vec
# ----------------------------------------------------------------------------------------------------------------------
def plt_signal(time_vec,X):
    plt.figure(figsize=(6, 5))
    plt.plot(time_vec, X, label='Original signal')
    plt.savefig('./data/output/signal.png')
    return
# ----------------------------------------------------------------------------------------------------------------------
def plt_FFT_power(sample_freq, power,frequency_based=True,filename_out = None):

    plt.figure(figsize=(6, 5))
    if frequency_based:
        plt.plot(sample_freq, power)
        #plt.xlim(left=0,right=0.5)
        plt.xlabel('Frequency [Hz]')
    else:
        periods = 1/sample_freq
        plt.plot(periods, power)
        #plt.xlim(left=0,right=20)
        plt.xlabel('Period')

    plt.ylabel('plower')
    if filename_out is not None:
        plt.savefig(filename_out)
    return
# ----------------------------------------------------------------------------------------------------------------------
def get_powerful_freq(sample_freq, power):
    idx = numpy.argsort(-power)
    return numpy.abs(sample_freq[idx[0]])
# ----------------------------------------------------------------------------------------------------------------------
time_step = 1
duration = 1000
# ----------------------------------------------------------------------------------------------------------------------
A = [2,1]
periods = numpy.array([10,13])
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #time_vec = generate_time_vec(duration,time_step)
    #X = tools_signal.generate_signal(time_vec,A,1/periods)

    #X = tools_IO.load_mat('./data/ex_fft/signal.txt',delim=' ',dtype=numpy.int)
    #time_vec = numpy.arange(0,len(X),1)

    image = tools_signal.generate_signal_2d(240,320,[100],[1.0/50])
    cv2.imwrite(folder_out+'image_signal.png',image)
    X = image[0,:]
    time_vec = numpy.arange(0,len(X),1)

    plt_signal(time_vec, X)
    sample_freq, power = tools_signal.get_AF(X)
    plt_FFT_power(sample_freq, power,frequency_based=False,filename_out=folder_out+'power.png')


