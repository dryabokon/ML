# ----------------------------------------------------------------------------------------------------------------------
import numpy
import pyaudio
import tools_audio
import progressbar
import tools_IO
import tools_video
import time
from playsound import playsound
# ----------------------------------------------------------------------------------------------------------------------
form_1 = pyaudio.paInt16
chans = 1
samp_rate = 44100
chunk = 4096
# ----------------------------------------------------------------------------------------------------------------------
def get_next_filename_out(base_folder_out,list_of_masks):

    filenames = tools_IO.get_filenames(base_folder_out,list_of_masks)
    filenames = [filename.split('.')[0] for filename in filenames]
    filenames = [filename for filename in filenames if filename.isdigit()]

    ext = list_of_masks.split(',')[0].split('.')[1]
    if len(filenames) > 0:
        filenames = numpy.sort(filenames)
        digit = filenames[-1].split('.')[0]
        if digit.isdigit():
            filename_out = str(int(digit)+1)
        else:
            filename_out='0'

        filename_out+='.'+ext
    else:
        filename_out = '0.'+ext
    return base_folder_out + filename_out

# ----------------------------------------------------------------------------------------------------------------------
def write_loop_current_sound_wav(filename_out,dev_index,duration=150):

    audio = pyaudio.PyAudio()
    stream = audio.open(format = form_1,rate = samp_rate,channels = chans, input_device_index = dev_index,input = True, frames_per_buffer=chunk)
    frames = []

    bar = progressbar.ProgressBar(max_value=duration)

    for b in range(duration):
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
def write_loop_current_sound_bin(filename_out,dev_index=5):

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
#filenames = tools_IO.get_filenames('./data/ex_wave/us/','*.wav')
#full_filenames = ['./data/ex_wave/us/'+each for each in filenames]
#tools_audio.merge_audio_files(full_filenames,'./data/output/us.wav')
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './data/output_sounds/'
# ----------------------------------------------------------------------------------------------------------------------
p = pyaudio.PyAudio()
for ii in range(p.get_device_count()): print(p.get_device_info_by_index(ii).get('name'))
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #write_loop_current_sound_bin(folder_out+'sound.dat')

    index = 0
    filename_out = get_next_filename_out(folder_out,'*.wav')
    write_loop_current_sound_wav(filename_out,index)





