
# Versions deployed for this software development
# Please if you want to know yours, use the version_test() method
#
# Versions: SciPy = 0.13.3; NumPy = 1.6.1; IPython = 0.12.1
# Python interpreter:
# 2.7.3 (default, Feb 27 2014, 19:58:35) 
# [GCC 4.6.3]
#
# Please if you want to know yours, use the version_test() method


# jack_control start
import copy
import pyaudio
import wave
import sys 
import scipy
import IPython

from array import array
from struct import pack
from sys import byteorder

import numpy as np
from matplotlib import mlab, pyplot
from collections import defaultdict


from scipy import signal, fft, arange, ifft
from scipy.signal import argrelmax
from scipy.io import wavfile

from collections import Counter

 
from numpy import sin, linspace, pi



#General Parameters
THRESHOLD = 3500 
CHUNK_SIZE = 1024
SILENT_CHUNKS = 3 * 44100 / 1024  # about 3sec
FORMAT = pyaudio.paInt16
FRAME_MAX_VALUE = 2 ** 15 - 1
NORMALIZE_MINUS_ONE_dB = 10 ** (-1.0 / 20)
RATE = 16000 #RATE = 44100
CHANNELS = 1 #1 = mono
TRIM_APPEND = RATE / 4
NFFT = 1024       # the length of the windowing segments

RECORD_SECONDS = 30



def normalize(data_all):
    """Amplify the volume out to max -1dB"""
    # MAXIMUM = 16384
    normalize_factor = (float(NORMALIZE_MINUS_ONE_dB * FRAME_MAX_VALUE)
                        / max(abs(i) for i in data_all))

    r = array('h')
    for i in data_all:
        r.append(int(i * normalize_factor))
    return r

def trim(data_all):
    _from = 0
    _to = len(data_all) - 1
    for i, b in enumerate(data_all):
        if abs(b) > THRESHOLD:
            _from = max(0, i - TRIM_APPEND)
            break

    for i, b in enumerate(reversed(data_all)):
        if abs(b) > THRESHOLD:
            _to = min(len(data_all) - 1, len(data_all) - 1 - i + TRIM_APPEND)
            break
    #print "showing #: %d %d\n" % (_from, _to )
    return copy.deepcopy(data_all[_from:(_to + 1)])

def check_alarm(ring_sample, freqs_array, threshhold):
    
    v_alarm_counter = 0
    for alarm_group in ring_sample:
      for freqs_list in freqs_array:
        if alarm_group-threshhold < freqs_list and alarm_group+threshhold > freqs_list:
          v_alarm_counter += 1
          
    return v_alarm_counter
    
def record():
    """Record a word or words from the microphone and 
    return the data as an array of signed shorts."""

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, output=True, frames_per_buffer=CHUNK_SIZE)

    audio_started = False
    data_all = array('h')
    total_freq = {}
    total_freq_val = 1
    
    DEBUG_ON   = 0
    DEBUG_ON_1 = 0
    time_ring_bell = 0
    #Fs = 48000
    Fs = 16000 # ESTABA A 16000
    f = np.arange(1, 9) * 2000
    t = np.arange(RECORD_SECONDS * Fs) / Fs 
    x = np.empty(t.shape)
    for i in range(8):
        x[i*Fs:(i+1)*Fs] = np.cos(2*np.pi * f[i] * t[i*Fs:(i+1)*Fs])

    w = np.hamming(512)
    #print np.hamming(512)[1:-1]
    #print len(np.hamming(512))
    RING_STATUS = 0
    NEVERA_STATUS = 0

    #while True:
    for i in range(0, int(RATE / CHUNK_SIZE * RECORD_SECONDS)):
        
        # little endian, signed shortdata_chunk
        data_chunk = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            data_chunk.byteswap()
        
        # add new kchun
        data_all.extend(data_chunk)

        # Get the data with specgram
        Pxx, freqs, bins = mlab.specgram(data_chunk, NFFT=512, Fs=Fs, window=w, noverlap=8*60)
        #, 
        #noverlap=8*60.4234324)
        #464 FRom 52 to 63
        
        if DEBUG_ON:
          print ("<<<<< RT >>>>>")
          print len(Pxx)
          print len(freqs)
          print len(bins)
        
        #print len(Pxx)
        #The Pxx value is a periodogram matrix, I get a slice in the column five 
        # we have 10
        periodogram_of_interest = Pxx[:, 5]
        
        #This is a temporary variable for counting the occurences of a given 
        # freq in a given periodogram
        v_alarm_test = 0
        
        ring_bell_training_array = [906, 2375, 4375, 6875]
        war_bell_training_array  = [968, 1937, 2906, 3875, 4843, 5812, 6781, 7750]
        wake_up_training_array   = [2062, 4093, 6156, 7812]
        doorbell_id_step_1       = [625, 1687, 3343, 5593]
        doorbell_id_step_2       = [1343, 3700, 4750, 6700 ]
	initial_training_array   = [562 ,  1687, 3000 ]
	nevera_warning_array     = [3000,    4031,  5031,  6031, 6475]
	horno_warning_array      = [2025,    4050,    5175,    6042]

        try:
        # we get the Local Maximum freqs in the freqs list
          v_freqs_list = freqs[argrelmax(periodogram_of_interest, order=20)]
        except IndexError:
          print IndexError
          continue

        if DEBUG_ON_1:
          print ' Frame id ' + str(i)
          print v_freqs_list
          
        v_alarm_test = check_alarm(ring_bell_training_array, v_freqs_list, 50)
        
        
        if v_alarm_test >= len(ring_bell_training_array):
          print ("******************************************************RING BELL")
          time_ring_bell += 1
        
        
        v_alarm_test = check_alarm(initial_training_array, v_freqs_list, 20)
       
        if v_alarm_test >= ((len(initial_training_array) - 1) ):
          print v_alarm_test
          print ("======================================  initial_training_array")
          
        
        v_alarm_test = check_alarm(doorbell_id_step_1, v_freqs_list, 20)
        #print ("v_alarm_counter ring bell 1: " , v_alarm_test)
       
        if v_alarm_test >= ((len(doorbell_id_step_1) - 1) ):
          if DEBUG_ON:
            print ("==============================================A")
          RING_STATUS = 1
        
        v_alarm_test = check_alarm(doorbell_id_step_2, v_freqs_list, 100)
        #print ("v_alarm_counter ring bell 2: " , v_alarm_test)

        if v_alarm_test >= ((len(doorbell_id_step_2) - 1) ) and RING_STATUS == 1:
          print ("==============================================DoorBell")
          RING_STATUS = 0
          

	#=======================================================================
	# checking home appliance
	# TODO 
	#=======================================================================

		
        v_alarm_test = check_alarm(nevera_warning_array, v_freqs_list, 50)
        #print ("v_alarm_counter nevera 1: " , v_alarm_test)
       
        if v_alarm_test >= ((len(nevera_warning_array) - 1)) and NEVERA_STATUS == 0:
          print ("============================================== Nevera Warning")
          NEVERA_STATUS = 1


  	v_alarm_test = check_alarm(horno_warning_array, v_freqs_list, 30)
        #print ("v_alarm_counter horno 1: " , v_alarm_test)
       
        if v_alarm_test >= ((len(horno_warning_array)) -1 )and NEVERA_STATUS == 0:
          print ("============================================== Horno Warning")
          

	if NEVERA_STATUS == 1:
	  NEVERA_STATUS = 0

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()
    info_debug(data_all, Fs, 0)

    #data_all = trim(data_all)  # we trim before normalize as threshhold applies to un-normalized wave (as well as is_silent() function)
    #data_all = normalize(data_all)
    return sample_width, data_all

    
    
#=======================================================================
# For checking files
# TODO 
#=======================================================================
    
def check_file():
    """Record a word or words from the microphone and 
    return the data as an array of signed shorts."""

    #p = pyaudio.PyAudio()
    #stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, output=True, frames_per_buffer=CHUNK_SIZE)

    data_all = array('h')
    
    DEBUG_ON = 1
    DEBUG_ON_1 = 1
        
    Fs = 16000 # ESTABA A 16000
    f = np.arange(1, 9) * 2000
    t = np.arange(RECORD_SECONDS * Fs) / Fs 
    x = np.empty(t.shape)
    for i in range(8):
        x[i*Fs:(i+1)*Fs] = np.cos(2*np.pi * f[i] * t[i*Fs:(i+1)*Fs])

    w = np.hamming(512)
    
    # open up a wave
    wf = wave.open('demo.wav', 'rb')
    seconds_width = wf.getsampwidth()
    #print (">>>>> tamano", seconds_width)
    RATE = wf.getframerate()
    #print (">>>>>Frame rate", RATE)

    # use a Blackman window
    window_tmp = np.blackman(CHUNK_SIZE)

    

    # read some data
    data_chunk_tmp = wf.readframes(CHUNK_SIZE)
    data_chunk = array('h', wf.readframes(CHUNK_SIZE))
    
    # play stream and find the frequency of each chunk
    while len(data_chunk_tmp) == CHUNK_SIZE * seconds_width:
                
        # little endian, signed shortdata_chunk
        #data_chunk = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            data_chunk.byteswap()
        
        # add new kchun
        data_all.extend(data_chunk)

        # Get the data with specgram
        #Pxx, freqs, bins = wavToSpec("demo.wav")
        # add new kchun
        data_all.extend(data_chunk)

        # Get the data with specgram
        Pxx, freqs, bins = mlab.specgram(data_chunk, NFFT=512, Fs=Fs, window=w, 
                                    noverlap=464)
        
        
        
        if DEBUG_ON:
          print ("<<<<< RT >>>>>")
          print len(Pxx)
          print len(freqs)
          print len(bins)
        
        
        #The Pxx value is a periodogram matrix, I get a slice in the column five 
        # we have 10
        periodogram_of_interest = Pxx[:, 5]
        
        #This is a temporary variable for counting the occurences of a given 
        # freq in a given periodogram
        v_alarm_test = 0
        
        ring_bell_training_array = [906, 2375, 4375, 6875]
        war_bell_training_array  = [968, 1937, 2906, 3875, 4843, 5812, 6781, 7750]
        wake_up_training_array   = [2062, 4093, 6156, 7812]
        doorbell_id_step_1       = [625, 1687, 3343, 5593]
        doorbell_id_step_2       = [1343, 3700, 4750, 6700 ]

        try:
          # we get the Local Maximum freqs in the freqs list
          v_freqs_list = freqs[argrelmax(periodogram_of_interest, order=20)]
        except IndexError:
          print IndexError
          continue
        
        
        if DEBUG_ON_1:
          print v_freqs_list
          
        v_alarm_test = check_alarm(ring_bell_training_array, v_freqs_list, 50)
        
        print v_extra_var
        if v_alarm_test >= len(ring_bell_training_array):
          print ("******************************************************RING BELL")
          time_ring_bell += 1
        
        
        v_alarm_test = check_alarm(war_bell_training_array, v_freqs_list, 20)
       
        if v_alarm_test >= ((len(war_bell_training_array) - 1) ):
          print ("==============================================WAR ALARM")
          
        
        v_alarm_test = check_alarm(doorbell_id_step_1, v_freqs_list, 20)
        #print ("v_alarm_counter ring bell 1: " , v_alarm_test)
       
        if v_alarm_test >= ((len(doorbell_id_step_1) - 1) ):
          if DEBUG_ON:
            print ("==============================================A")
          RING_STATUS = 1
        
        v_alarm_test = check_alarm(doorbell_id_step_2, v_freqs_list, 100)
        #print ("v_alarm_counter ring bell 2: " , v_alarm_test)

        if v_alarm_test >= ((len(doorbell_id_step_2) - 1) ) and RING_STATUS == 1:
          print ("==============================================DoorBell")
          RING_STATUS = 0
          
        data_chunk_tmp = wf.readframes(CHUNK_SIZE)


    
    info_debug(data_all, Fs, 1)

    #data_all = trim(data_all)  # we trim before normalize as threshhold applies to un-normalized wave (as well as is_silent() function)
    #data_all = normalize(data_all)
    return  
    
def wavToSpec(wavefile,log=False,norm=False):
    wavArr,wavParams = wavToArr(wavefile)
    print wavParams
    return  mlab.specgram(wavArr, NFFT=256,Fs=wavParams[2],window=mlab.window_hanning,noverlap=128,sides='onesided',scale_by_freq=True)

def wavToArr(wavefile):
    w = wave.open(wavefile,"rb")
    p = w.getparams()
    s = w.readframes(p[3])
    w.close()
    sd = np.fromstring(s, np.int16)
    return sd,p
    
    
#=======================================================================
# Debug INFO  
#=======================================================================
    
def info_debug(data_all,Fs, mode):

    if mode == 0:
      Pxx, freqs, bins = mlab.specgram(data_all, NFFT=512, Fs=Fs, window=np.hamming(512), 
                                    noverlap=464)
    else :
      Pxx, freqs, bins = wavToSpec("demo.wav")

    C_DEBUG_ON      = 0
    C_DEBUG_ON_1    = 1
    C_DEBUG_FILE_ON = 0
    
    
    temporal_a   = 0
    temp_values  = []
    temp = np.zeros(8000)
    
    if C_DEBUG_ON:
      print ("<<<<< 1 >>>>>")
      print Pxx
      print freqs
      print bins
      
    for wav_f in range(0,len(bins),len(bins)/100):
     
      periodogram_of_interest = Pxx[:, wav_f]
     
      
      #print periodogram_of_interest
      try:
        temp_values = argrelmax(periodogram_of_interest, order=20)
     
      
        if C_DEBUG_ON:
          print ("<<<<< 2 >>>>>")
          print temp_values
          print freqs[temp_values]
          
        for recol in freqs[temp_values]:
          if temp[int(recol)] == 0:
            temp [int(recol)] = 1
          else :
            temp [int(recol)] += 1
        
        if C_DEBUG_ON:
          print temp [int(recol)]
        
      except IndexError:
        print IndexError
        return
        
    #If we want to print all the values saved    
    if C_DEBUG_ON:
      t_var1 = 0
      for t_var1 in range(len(temp)):
        if temp[t_var1] != 0.:
          print (t_var1,  temp[t_var1])

    
    if C_DEBUG_ON:
      print ("<<<<< Lengh: Pxx, freqs, bins >>>>>")
      print len(Pxx)
      print len(freqs)
      print len(bins)
     
      print ("<<<<< Argrelmax values >>>>>")
      print argrelmax(Pxx)
      print argrelmax(freqs)
      print argrelmax(bins)

    #=======================================================================
    # plot the spectrogram in dB
    #=======================================================================

    Pxx_dB = np.log10(Pxx)
    pyplot.subplots_adjust(hspace=0.3)

    pyplot.subplot(411)
    ex1 = bins[0], bins[-1], freqs[0], freqs[-1]
    pyplot.imshow(np.flipud(Pxx_dB), extent=ex1)
    #pyplot.axis('auto')
    pyplot.axis('tight')
    pyplot.xlabel('time (s)')
    pyplot.ylabel('freq (Hz)')

    
    #print ("The max number is  >>>>>", np.max(Pxx), " - ",  np.max(bins))
    #Pxx_dB = np.log10(Pxx)
    #print ("The max number is  >>>>>", np.max(Pxx_dB))
   
    if C_DEBUG_FILE_ON:
      np.savetxt("./temp_PXX", Pxx, fmt = '%f')
      np.savetxt("./temp_PXX_db", Pxx_dB, fmt = '%f')
    
    #=======================================================================
    # plot the real spectrogram
    #=======================================================================
    #pyplot.subplot(412)
    #ex1 = bins[0], bins[-1], freqs[0], freqs[-1]
    #pyplot.pcolormesh(bins, freqs, Pxx)
    #pyplot.axis('auto')
    #pyplot.axis(ex1)
    #pyplot.xlabel('time (s)')
    #pyplot.ylabel('freq (Hz)')
    
    #=======================================================================
    #====== We plot the real sin signal
    #=======================================================================
    #====== Method 1
    y=data_all[:]
    lungime=len(y)
    timp=len(y)/RATE
    t=linspace(0,timp,len(y))

    pyplot.subplot(412)
    pyplot.plot(t,y)
    pyplot.xlabel('Time')
    pyplot.ylabel('Amplitude')
    
    #====== Method 2
    #pyplot.subplot(413)
    ##pyplot.plot(range(len(data_all)),data_all)
    #pyplot.plot(range(len(data_all)),data_all)
    ##ex1 = bins[0], bins[-1], -1, 1
    #pyplot.axis('auto')
    #pyplot.xlabel('time (s)')
    #pyplot.ylabel('amp ')
    
    #=======================================================================
    #  We plot the spectrum in Y mode
    #=======================================================================
    pyplot.subplot(413)
    plotSpectru(y,Fs)
    
    #=======================================================================
    #  We plot the occurences of each freq / 2
    #=======================================================================
    pyplot.subplot(414)
    pyplot.xlabel('Freq (Hz)')
    pyplot.ylabel('Total (counts)')
    #pyplot.plot(*zip(*testList2))
    pyplot.plot(range(len(temp)),temp)

       
    pyplot.show()

    return 

    
def plotSpectru(y,Fs):
    n = len(y) # lungime semnal
    k = arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(n/2)] # one side frequency range

    Y = fft(y)/n # fft computing and normalization
    Y = Y[range(n/2)]
    
    pyplot.plot(frq,abs(Y),'r') # plotting the spectrum
    pyplot.xlabel('Freq (Hz)')
    pyplot.ylabel('|Y(freq)|')
    
def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    data = pack('<' + ('h' * len(data)), *data)

    wave_file = wave.open(path, 'wb')
    wave_file.setnchannels(CHANNELS)
    wave_file.setsampwidth(sample_width)
    wave_file.setframerate(RATE)
    wave_file.writeframes(data)
    wave_file.close()

def version_test():
    #print("Versions: SciPy = {}; NumPy = {}; IPython = {}".format(scipy.__version__, np.__version__, IPython.__version__))
    #print 'Python interpreter: ' + sys.version
    
    major,minor,sub = scipy.__version__.split('.')[:3]
    if int(major) == 0 and int(minor) < 11:
      print 'Please Update de SciPy Library Please'

    
if __name__ == '__main__':
  
    version_test()
    
    #print 'Number of arguments:', len(sys.argv), 'arguments.'
    #print 'Argument List:', str(sys.argv)
    print '****************************************'
    print '  Welcome to RingBell Modelling system  '
    print '****************************************'
    
    if len(sys.argv) == 2:
      print '*** This is a work in progress *** '
      #check_file()
    else :
      record_to_file('demo.wav')
    
    print '***************************************'
    print '  done - result written to demo.wav    '
    print '  See you soon'
    print '***************************************'
    
