# ----------------------------------------------------------------------------------------------------------------------
import numpy
import pyaudio
import wave
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
def save_steam(filename_out,frames,audio,chans = 1,form_1=pyaudio.paInt16,samp_rate=44100,):
    with wave.open(filename_out, 'wb') as wavefile:
        wavefile.setnchannels(chans)
        wavefile.setsampwidth(audio.get_sample_size(form_1))
        wavefile.setframerate(samp_rate)
        wavefile.writeframes(b''.join(frames))
        wavefile.close()
    return
# ----------------------------------------------------------------------------------------------------------------------
def record_wav(folder_out,record_secs,dev_index=0):
    p = pyaudio.PyAudio()
    #for ii in range(p.get_device_count()):
        #print(p.get_device_info_by_index(ii).get('name'))

    form_1 = pyaudio.paInt16
    chans = 1
    samp_rate = 44100
    chunk = 4096
    audio = pyaudio.PyAudio()
    stream = audio.open(format = form_1,rate = samp_rate,channels = chans, input_device_index = dev_index,input = True, frames_per_buffer=chunk)

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    data_array = numpy.zeros(chunk)
    data_array[0] = 10000
    data_array[1] = -data_array[0]
    line1, = ax.plot(data_array)
    frames = []

    for ii in range(0, int((44100 / 4096) * record_secs)):
        data = stream.read(4096)
        data_array = numpy.fromstring(stream.read(chunk), dtype=numpy.int16)
        frames.append(data)

        line1.set_ydata(data_array)
        fig.canvas.draw()
        fig.canvas.flush_events()

        #f_handle = open(folder_out + "data.txt", "wb")
        #f_handle.write(data)
        #f_handle.close()

    stream.stop_stream()
    stream.close()
    audio.terminate()
    save_steam(folder_out+'sound.wav',frames,audio)

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    record_wav('./data/output/',20)


