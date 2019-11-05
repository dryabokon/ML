# ----------------------------------------------------------------------------------------------------------------------
import numpy
import pyaudio
import tools_audio
import progressbar
from playsound import playsound
# ----------------------------------------------------------------------------------------------------------------------
form_1 = pyaudio.paInt16
chans = 1
samp_rate = 44100
chunk = 4096
# ----------------------------------------------------------------------------------------------------------------------
def write_loop_current_sound_wav(filename_out,dev_index=1):

    audio = pyaudio.PyAudio()
    stream = audio.open(format = form_1,rate = samp_rate,channels = chans, input_device_index = dev_index,input = True, frames_per_buffer=chunk)
    frames = []

    N = 200
    bar = progressbar.ProgressBar(max_value=N)

    for b in range(N):
        bar.update(b)
        data = stream.read(chunk)
        frames.append(data)

    tools_audio.save_steam_wav(filename_out, frames, audio)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    #playsound(filename_out, False)
    return
# ----------------------------------------------------------------------------------------------------------------------
def write_loop_current_sound_bin(filename_out,dev_index=1):

    audio = pyaudio.PyAudio()
    stream = audio.open(format = form_1,rate = samp_rate,channels = chans, input_device_index = dev_index,input = True, frames_per_buffer=chunk)

    while True:
        data = stream.read(chunk)
        f_handle = open(filename_out, "wb")
        f_handle.write(numpy.fromstring(data, dtype=numpy.int16))
        f_handle.close()

    stream.stop_stream()
    stream.close()
    audio.terminate()

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #write_loop_current_sound_bin('./data/output/sound.dat')
    write_loop_current_sound_wav('./data/output/sound.wav')
    #p = pyaudio.PyAudio()
    #for ii in range(p.get_device_count()): print(p.get_device_info_by_index(ii).get('name'))

