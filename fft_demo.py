# Created on 8/23/22 at 12:47 PM 

# Author: Jenny Sun
'''
this script is meant to demonstrate FFT
with numpy fft function
'''
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft

####### test FFT algorithm ##########
# sampling rate
sr = 500

# sampling interval
dur = 2     # 2 s
L = sr *dur   # signal length
T = 1.0/sr # sampling period
t = np.arange(0,L) * T  # time vector

# make the signals
freq0 = 1
x= []
x = 3*np.sin(2*np.pi*freq0*t)

freq1 = 4
x += np.sin(2*np.pi*freq1*t)

freq2 = 10
x += 0.5* np.sin(2*np.pi*freq2*t)

plt.figure(figsize = (8, 6))
plt.plot(t, x, 'r')
plt.ylabel('Amplitude')
plt.show()


# fft the data
Y = fft(x)
# compute the two-sided spectrum P2
P2 = np.abs(Y/L) #power spectrum

# then compute the single-sided spectrum P1 based on P2a and the real-valued signal length L
P1 = P2[0:int(L/2)]  # 0 Hz included
P1[1:] = 2 * P1[1:]

# define the frequency domain f
f = sr * np.arange(0, L/2) /L

plt.figure()
plt.plot(f, P1)
plt.xlim(0,20)
plt.show()

# let's inverse FFT

