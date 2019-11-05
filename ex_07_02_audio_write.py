# ----------------------------------------------------------------------------------------------------------------------
import numpy
import pyaudio
import wave
# ----------------------------------------------------------------------------------------------------------------------
form_1 = pyaudio.paInt16
chans = 1
samp_rate = 44100
chunk = 4096
# ----------------------------------------------------------------------------------------------------------------------
def save_steam_wav(filename_out,frames,audio,chans = 1,form_1=pyaudio.paInt16,samp_rate=44100,):
    with wave.open(filename_out, 'wb') as wavefile:
        width = audio.get_sample_size(form_1)
        wavefile.setnchannels(chans)
        wavefile.setsampwidth(width)
        wavefile.setframerate(samp_rate)
        wavefile.writeframes(b''.join(frames))
        wavefile.close()
    return
# ----------------------------------------------------------------------------------------------------------------------
def write_loop_current_sound_wav(filename_out,dev_index=1):

    audio = pyaudio.PyAudio()
    stream = audio.open(format = form_1,rate = samp_rate,channels = chans, input_device_index = dev_index,input = True, frames_per_buffer=chunk)

    while True:
        frames = []
        while len(frames)<100:
            data = stream.read(chunk)
            frames.append(data)

        save_steam_wav(filename_out, frames, audio)

        #f_handle = open('./data/output/sound.dat', "wb")
        #f_handle.write(numpy.fromstring(data, dtype=numpy.int16))
        #f_handle.close()

        print('.')

    stream.stop_stream()
    stream.close()
    audio.terminate()

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


'''
Microsoft Sound Mapper - Input
Headset Microphone (Jabra UC VO
Microphone (High Definition Aud
Microsoft Sound Mapper - Output
Headset Earphone (Jabra UC VOIC
Speakers (High Definition Audio
'''