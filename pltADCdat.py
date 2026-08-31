#!/home/anish/anaconda3/bin/python3

import socket
import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os

Nsamp=61.44 #MHz
#Nsamp=245.76 #MHz
packetsize=8256 #bytes of data in a packet including header
headersize=64  #bytes of header; first 64bit is counter
datsize=packetsize-headersize #bytes from 4 ADCs

ADC=['ADC0', 'ADC1', 'ADC2', 'ADC3']
Datdir = '/mnt/carsedat/'
#Input to specify the directory
#filetype='20240816_141152_0000.dat'
#filetype='20241004_164528_0000.dat'
#filetype='20250306_161313_0000.dat'
#filetype='20250625_205729_0000.dat'
#filetype='20250625_210107_0000.dat'
#filetype='20250625_210455_0000.dat'
#filetype='20250702_202627_0000.datnp0'
#filetype='20250718_152636_0000.dat'
#filetype='20250909_173648_0000.dat'

#filetype='20251209_192403_0000.datnp1'
#filetype='20251209_192821_0000.datnp1'
#filetype='20251209_194523_0000.datnp0'
#filetype='20251209_194712_0000.datnp1'

#noise floor analysis of signal chain + ADC
filetype='20260317_185653_0000.dat'

ADCdatperpacket=1024 #init16 values per ADC
NFFT=ADCdatperpacket #int16 values per ADC

print("Number of point FFT : ", NFFT)

sintval=np.zeros(datsize//2*NFFT//ADCdatperpacket, dtype=np.int16)
count=np.zeros(NFFT//ADCdatperpacket, dtype=np.uint64)


file_path=Datdir+filetype
if os.path.exists(file_path):
   with open(file_path, "rb") as file:
     for cnt in np.arange(NFFT//ADCdatperpacket):
       hdat = np.frombuffer(file.read(headersize), dtype=np.uint64)
       dat = np.frombuffer(file.read(datsize), dtype=np.int16)
       sintval[(cnt*datsize//2):((cnt+1)*datsize//2)]=dat
       count[cnt]=hdat[0]
else:
   print(Datdir+filetype, " file does not exist!!")
   sys.exit(1)

fig1, ax = plt.subplots(nrows=2, ncols=1,constrained_layout=True)
fig1.set_figwidth(6)
fig1.set_figheight(8)

params = {
   'axes.labelsize': 15,
   'font.size': 15,
   'legend.fontsize': 12,
   'xtick.labelsize': 15,
   'ytick.labelsize': 15,
   'text.usetex': False,
   'figure.figsize': [10, 6],
   'lines.linewidth': 3
   }
mpl.rcParams.update(params)

for axis in ['top','bottom','left','right']:
    ax[0].spines[axis].set_linewidth(2)
    ax[1].spines[axis].set_linewidth(2)

ax[0].tick_params(width=2)
ax[1].tick_params(width=2)

#plot time series and power spec
for adccnt in np.arange(4):
   adcdat=sintval[adccnt::4]
   ax[0].plot(adcdat, label=ADC[adccnt], linewidth=3)
   Ps=np.abs(fft(adcdat))
   freq=np.arange(len(Ps))*Nsamp/len(Ps)
   ax[1].plot(freq, 20*np.log10(Ps/Ps.max()), label=ADC[adccnt], linewidth=3)

if sum(np.diff(count)) != len(np.diff(count)):
   print("**************** Jump in count ************")
   print(count)

plt.savefig('temp.png')
plt.show()

