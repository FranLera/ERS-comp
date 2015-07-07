#!/usr/bin/env python

NODE_NAME = 'bell_recognition'

import roslib; roslib.load_manifest(NODE_NAME)

import os
import rospy
#audio record start
import pyaudio
import numpy as np


from std_msgs.msg import String
from sys import byteorder
from matplotlib import mlab, pyplot
from scipy.signal import argrelmax
# array reserved element
from array import array


#GENERAL CONSTANTS

CHUNK_SIZE = 1024
SILENT_CHUNKS = 3 * 44100 / 1024  # about 3sec
FORMAT = pyaudio.paInt16
FRAME_MAX_VALUE = 2 ** 15 - 1
NORMALIZE_MINUS_ONE_dB = 10 ** (-1.0 / 20)
#RATE = 44100
RATE = 16000
CHANNELS = 1
TRIM_APPEND = RATE / 4

NFFT = 1024       # the length of the windowing segments

C_DEBUG_ON = 0
C_DEBUG_ON_1 = 0
C_FS = 16000

# method for check the bell
# ring_sample: we have to define the main frequencies of the signal
# freqs_array: This parameter contains the main frequencies in a slice of time (Maximum Locals)
# threshold : This parameter allows to work with a threshold of frequencies. This happens when 
#             the microphone is changed or the mic characteristics

def check_alarm(ring_sample, freqs_array, threshold):
    
  v_alarm_counter = 0
  for alarm_group in ring_sample:
    for freqs_list in freqs_array:
      if alarm_group-threshold < freqs_list and alarm_group+threshold > freqs_list:
        v_alarm_counter += 1
        
  return v_alarm_counter
  
    
        
def start():


  pub = rospy.Publisher('/bell_recog', String)

  p = pyaudio.PyAudio()
  stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, 
                  output=True, frames_per_buffer=CHUNK_SIZE)

  rospy.loginfo("Running.....")

  silent_chunks = 0
  audio_started = False
  data_all = array('h')
  total_freq = {}
  total_freq_val = 1

  ring_bell_init = 0
  ring_bell_clean = 0
  v_doorbell_status = 0
  last_ring_bell_init = 0

  v_fridge_status = 0
  hornovalue = 0
  while not rospy.is_shutdown():
  
      #If we want to test the byteorder
      # little endian, signed shortdata_chunk
      data_chunk = array('h', stream.read(CHUNK_SIZE))
      if byteorder == 'big':
        data_chunk.byteswap()

      Pxx, freqs, bins = mlab.specgram(data_chunk, NFFT=512, Fs=C_FS, 
                            window=np.hamming(512), noverlap=464)
     
      # Maybe we want to use this for time measurement
      now = rospy.get_rostime()
      #rospy.loginfo("Current time %i %i", now.secs, now.nsecs)
  
   
      periodogram_of_interest = Pxx[:, 5]
      
      if C_DEBUG_ON:
        print len(Pxx)
        print len(freqs)
        print len(bins)
        print Pxx
        print periodogram_of_interest
        print argrelmax(periodogram_of_interest, order=20)
        print isinstance(periodogram_of_interest[0], int)
        
      try:
        inNumberint = int(periodogram_of_interest[0])
        if C_DEBUG_ON:
          print freqs[argrelmax(periodogram_of_interest, order=20)]
      except ValueError:
        print ValueError
        
      
      alarmilla = 0
      avoid_record = 0
      #ring_bell_training_array = [2375, 4375, 6875]
      #ring_bell_training_array = [906, 2375, 4375, 6875]
      doorbell_id_step_1       = [625, 1687, 3343, 5593]
      doorbell_id_step_2       = [1343, 3700, 4750, 6700 ]
      ring_bell_training_array = [968, 1937, 2906, 3875, 4843, 5812, 6781, 7750]
      #nevera_warning_array     = [3000,    4031,  5031,  6031, 6475]
      nevera_warning_array     = [812, 2000,    3000,  4000,    5000,    6000 ]
      horno_warning_array      = [2025,    4050,    5175,    6042]
      
        
      try:
        # We need a try-except expression for getting some problems related with 
        # hardware management
        
        # we get the Local Maximum freqs in the freqs list var
        v_freqs_list = freqs[argrelmax(periodogram_of_interest, order=20)]
        
        if C_DEBUG_ON:
          print v_freqs_list
          
      except IndexError:
        print IndexError
        #example = last_example
        continue

      #==============================================================
      # Check Ring bell (small bell from the lab)
      #==============================================================
      #v_alarm_test = check_alarm(ring_bell_training_array, v_freqs_list, 50)
      v_alarm_test = check_alarm(ring_bell_training_array, v_freqs_list, 100)
        
        
      if v_alarm_test >= len(ring_bell_training_array):
        rospy.loginfo("Ring bell")
        
        ring_bell_init = now.secs
	if last_ring_bell_init == 0:
	  last_ring_bell_init = ring_bell_init
      else :
        ring_bell_init = 0 


      if ring_bell_init == 0 and last_ring_bell_init > 0 :
	last_ring_bell_init = 0     
	pub.publish("WAR alarms")
    
      #==============================================================
      # Door bell from youtube: https://www.youtube.com/watch?v=HdGXGl19Tzk
      #==============================================================
      v_alarm_test = check_alarm(doorbell_id_step_1, v_freqs_list, 20)
       
      if v_alarm_test >= ((len(doorbell_id_step_1) - 1) ):
        v_doorbell_status = 1

        
      v_alarm_test = check_alarm(doorbell_id_step_2, v_freqs_list, 200)

      if v_alarm_test >= ((len(doorbell_id_step_2) - 1) ) and v_doorbell_status == 1:
        rospy.loginfo("DoorBell")
        pub.publish("DOOR")
        v_doorbell_status = 0  
      
      #=======================================================================
      # checking home appliance
      # TODO: change v_fridge_status for a time variable
      #=======================================================================

                
      v_alarm_test = check_alarm(nevera_warning_array, v_freqs_list, 50)
      # DEbug purpose
      #print ("v_alarm_counter nevera 1: " , v_alarm_test)
      
      if v_alarm_test >= ((len(nevera_warning_array) - 1)) and v_fridge_status == 0:
        v_fridge_status = 1
        hornovalue = 0
      elif v_alarm_test >= ((len(nevera_warning_array) - 1)) and v_fridge_status == 1:
        v_fridge_status = 2
      elif v_alarm_test >= ((len(nevera_warning_array) - 1)) and v_fridge_status == 2:
        v_fridge_status = 3
      elif v_alarm_test >= ((len(nevera_warning_array) - 1)) and v_fridge_status == 3:
        rospy.loginfo("FRIDGE")
        pub.publish("FRIDGE")
        v_fridge_status = 0

      v_alarm_test = check_alarm(horno_warning_array, v_freqs_list, 30)
      print ("v_alarm_counter horno 1: " , v_alarm_test)
      
      if v_alarm_test >= ((len(horno_warning_array)) - 1 )  and hornovalue == 0:
        hornovalue = 1
        v_fridge_status = 0
      elif v_alarm_test >= ((len(horno_warning_array)) -1 ) and hornovalue == 1:  
        rospy.loginfo("OVEN")
        pub.publish("OVEN")
        hornovalue = 0
        
      
      
if __name__ == '__main__':
  
    print ("***********************************")
    print (" NODE STARTED  ")
    print ("***********************************")

    rospy.init_node(NODE_NAME, anonymous=True)
  
    start()
    
    print ("***********************************")
    print ("         NODE EXIT")
    print ("***********************************")
