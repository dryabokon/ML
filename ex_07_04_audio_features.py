#https://www.analyticsvidhya.com/blog/2017/08/audio-voice-processing-deep-learning/
# ----------------------------------------------------------------------------------------------------------------------
import os
import numpy
import pyaudio
from os import listdir
import fnmatch
from scipy import signal
import librosa
import tools_IO
import tools_ML
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from playsound import playsound
import tools_audio
# ----------------------------------------------------------------------------------------------------------------------
matplotlib.rcParams['toolbar'] = 'None'
# ----------------------------------------------------------------------------------------------------------------------
form_1 = pyaudio.paInt16
chans = 1
samp_rate = 44100
chunk = 4096
fs = 44110
# ----------------------------------------------------------------------------------------------------------------------
g_key=' '
g_click_data_x=-1
g_click_data_y=-1
g_click_flag=False
# ----------------------------------------------------------------------------------------------------------------------
def press(event):
    global g_key
    g_key = event.key
    return
# ----------------------------------------------------------------------------------------------------------------------
def onclick(event):
    global g_click_data_x,g_click_data_y,g_click_flag
    if event.button:
        g_click_data_x,g_click_data_y = event.xdata, event.ydata
        g_click_flag=True
    return
# ----------------------------------------------------------------------------------------------------------------------
def get_features_welch(X,fs,nperseg):
    f, Pxx_spec = signal.welch(X, fs, 'flattop', nperseg, scaling='spectrum')
    return numpy.sqrt(Pxx_spec)
# ----------------------------------------------------------------------------------------------------------------------
def get_features_mfcc(X,fs,nperseg):
    #fch = numpy.mean(librosa.feature.mfcc(X, sr=fs, n_mfcc=40).T, axis=0)
    fch = numpy.array(librosa.feature.mfcc(X, sr=fs,n_mfcc=40)).flatten()
    return fch
# ----------------------------------------------------------------------------------------------------------------------
def get_features_chroma_stft(X,fs,nperseg):
    fch = numpy.array(librosa.feature.chroma_stft(X, sr=fs)).flatten()
    return fch
# ----------------------------------------------------------------------------------------------------------------------
def get_features_mel(X,fs,nperseg):
    fch = numpy.array(librosa.feature.melspectrogram(X, sr=fs)).flatten()
    return fch
# ----------------------------------------------------------------------------------------------------------------------
def get_features_contrast(X,fs,nperseg):
    fch = numpy.array(librosa.feature.spectral_contrast(X, sr=fs)).flatten()
    return fch
# ----------------------------------------------------------------------------------------------------------------------
def get_features_tonnetz(X,fs,nperseg):
    fch = numpy.array(librosa.feature.tonnetz(X, sr=fs)).flatten()
    return fch
# ----------------------------------------------------------------------------------------------------------------------
def extract_features_from_sound_file(filename_in,filaname_out):
    X, fs = librosa.load(filename_in)
    X *= 0x7FFF
    features = []
    markup=[]

    step = 15 * chunk
    start, stop = 0, step

    while stop < len(X):
        start += int(step*3/4)
        stop  += int(step*3/4)

        feature = []
        feature.append(get_features_welch(X[start:stop], fs, chunk))
        #feature.append(get_features_mfcc(X[start:stop], fs, chunk))
        #feature.append(get_features_chroma_stft(X[start:stop], fs, chunk))
        #feature.append(get_features_mel(X[start:stop], fs, chunk))
        #feature.append(get_features_contrast(X[start:stop], fs, chunk))
        #feature.append(get_features_tonnetz(X[start:stop], fs, chunk))

        features.append(numpy.concatenate(feature))
        markup.append(filename_in+'_%d_%d'%(start,stop))


    if len(features[-1])!=len(features[-2]):
        features.pop(-1)
        markup.pop(-1)

    features = numpy.array(features)
    markup = numpy.array(markup)

    mat = numpy.zeros((features.shape[0], features.shape[1] + 1),dtype=numpy.chararray)
    mat[:, 0] = markup
    mat[:, 1:] = features
    tools_IO.save_mat(mat, filaname_out, fmt='%s', delim='\t')

    return
# ----------------------------------------------------------------------------------------------------------------------
def extract_features_from_sound_folder(folder_in,folder_out):
    all_filenames = tools_IO.get_filenames(folder_out,'*.*')
    filenames_in  = tools_IO.get_filenames(folder_in,'*.wav,*.mp3')
    filenames_out = [filename.split('.')[0] + '.txt' for filename in filenames_in]

    for all_filename in all_filenames:
        if all_filename not in filenames_out:
            tools_IO.remove_file(folder_out+all_filename)

    for filename_in,filename_out in zip(filenames_in,filenames_out):
        if not os.path.exists(folder_out + filename_out):
            extract_features_from_sound_file(folder_in+filename_in, folder_out + filename_out)

    print('extract_features_from_sound_folder OK')
    return
# ----------------------------------------------------------------------------------------------------------------------
def get_index(XY,x,y):

    delta = (XY[:,0]-x)**2 + (XY[:,1]-y)**2
    idx = numpy.argmin(delta)

    return idx
# ----------------------------------------------------------------------------------------------------------------------
def plot_features_PCA(folder_in, folder_out,has_header=True,has_labels_first_col=True):


    patterns = fnmatch.filter(listdir(folder_in), '*.txt')
    for i in range (0,len(patterns)):
        patterns[i]=patterns[i].split('.')[0]

    patterns = numpy.array(patterns)
    ML = tools_ML.tools_ML(None)

    X,Y,  filenames = ML.prepare_arrays_from_feature_files(folder_in, patterns=patterns, feature_mask='.txt',has_header=has_header,has_labels_first_col=has_labels_first_col)
    X_TSNE = TSNE(n_components=2).fit_transform(X)

    plt.ion()
    fig = plt.figure()
    fig.canvas.mpl_connect('key_press_event', press)
    fig.canvas.mpl_connect('button_press_event', onclick)

    fig.subplots_adjust(hspace =0.01)

    tools_IO.plot_2D_scores_multi_Y(plt.subplot(1,1,1),X_TSNE, Y, labels=patterns)

    plt.tight_layout()

    dict_sound,dict_fs = {},{}

    global g_click_flag

    i=0
    while True:
        fig.canvas.draw()
        fig.canvas.flush_events()

        if g_click_flag ==True:
            i+=1
            g_click_flag = False
            idx = get_index(X_TSNE,g_click_data_x,g_click_data_y)
            filename = filenames[idx].split('.')[1]
            ext = filenames[idx].split('.')[2].split('_')[0]
            start = filenames[idx].split('_')[-2]
            stop = filenames[idx].split('_')[-1]
            temp_file = folder_out + '%d.wav'%i
            #tools_audio.trim_file('.'+filename+'.'+ext,int(start),int(stop),temp_file)
            full_filename = '.' + filename + '.' + ext
            if full_filename not in dict_sound.keys():
                x, fs = librosa.load(full_filename)
                dict_sound[full_filename]=x
                dict_fs[full_filename]=fs

            x = dict_sound[full_filename]
            fs = dict_fs[full_filename]
            tools_audio.trim_X(x,fs,int(start),int(stop),temp_file)

            playsound(temp_file, False)



        if g_key == 'escape': return

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    folder_in  = './data/output_sounds/'
    folder_out = './data/output/'
    extract_features_from_sound_folder(folder_in,folder_out)
    plot_features_PCA(folder_out,folder_out,has_header=False,has_labels_first_col=True)
