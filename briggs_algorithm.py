#!/home/anish/anaconda3/bin/python3

import socket
import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import windows
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os
import argparse


#argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Briggs Algorithm")
    parser.add_argument("input_file", help="Input file containing noisy sine data")
    parser.add_argument("--FFT-size", type=int, default=1024, help="Size of the FFT window")
    parser.add_argument("--overlap", type=int, default=512, help="Overlap between FFT windows")
    parser.add_argument("--window", type=str, default='hamming', help="Window function to use for FFT segments")
    parser.add_argument("--plot-output", type=str, default=None, help="Output file to save the comparison plot")
    args = parser.parse_args()
    if args.plot_output is None:
        args.plot_output = 'comparison_spectrogram.png'
    return args

    return parser.parse_args()


def window_fft_data(data,fs, wind_len,overlap, window):

   #window building
   #compute step size for window
   step = wind_len - overlap 
   #find number of segments per window
   num_segments = (len(data) - overlap) // step
   if num_segments <= 0:
       raise ValueError("Signal duration is shorter than the segment size.")
   #make a 2d array of windowed data
   shape = (num_segments, wind_len)
   strides = (data.strides[0] * step, data.strides[0])
   segments = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
   win = windows.get_window(window, wind_len)
   windowed_segments = segments * win

   #remove mean (DC means nothing in this context)
   windowed_segments = windowed_segments - np.mean(windowed_segments, axis=1, keepdims=True)

   ## take FFT of windowed functions
   fft_segments = np.fft.rfft(windowed_segments, axis=-1)
   frequencies = np.fft.rfftfreq(wind_len, d=1/fs)
   return fft_segments, frequencies, win

def spectro_compare(P1, P1_corrected,freqs, plot_output):
    #make sure both are in dB + make sure no log of zero
    P1_dB = 10 * np.log10(P1 + 1e-12)
    P1_corrected_dB = 10 * np.log10(P1_corrected + 1e-12)
    #find common min/max for plot
    vmin = np.min([P1_dB, P1_corrected_dB])
    vmax = np.max([P1_dB, P1_corrected_dB])

    #make side by side by side comparison of the spectrograms
    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(10,8),sharex=True,sharey=True)
    im1 = ax1.imshow(P1_dB.T, aspect='auto', origin='lower', vmin=vmin, vmax=vmax, cmap='viridis')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Original Power Spectral Density')
    im2 = ax2.imshow(P1_corrected_dB.T, aspect='auto', origin='lower', vmin=vmin, vmax=vmax, cmap='viridis')
    ax2.set_title('Corrected Power Spectral Density')
    plt.colorbar(im1, ax=ax1)
    plt.colorbar(im2, ax=ax2)
    ax2.set_xlabel('Time')
    ax1.set_ylabel('Frequency')

    plt.tight_layout()
    if plot_output:
        plt.savefig(plot_output)
    plt.show()

def main():
    args = parse_args()
    Fsamp = 61.44e6
    # rest of the main function implementation
    # read the input file
    dat0,dat1,dat2,dat3 = np.genfromtxt(args.input_file, delimiter=',',dtype=np.int16,unpack=True)

    #compute windowed FFTs
    S1, freqs_0, win_0 = window_fft_data(dat0, Fsamp, args.FFT_size, args.overlap, args.window)
    S2, _, _ = window_fft_data(dat1, Fsamp, args.FFT_size, args.overlap, args.window)
    S3, _, _ = window_fft_data(dat2, Fsamp, args.FFT_size, args.overlap, args.window)
    S4, _, _ = window_fft_data(dat3, Fsamp, args.FFT_size, args.overlap, args.window)

    #compute power spectral density 
    P1 =(S1 * np.conj(S1)).real # power spectral density is only real values 

    #compute CSD for P1
    C13 = (S1 * np.conj(S3))
    C14 = (S1 * np.conj(S4))
    C34 = (S3 * np.conj(S4))

    #compute correction factor
    CX1 = (C13 * np.conj(C14)) / C34

    #correct the power spectral density using the correction factor
    P1_corrected = P1 - np.real(CX1)

    #plot both as spectrograms
    print("P1 Shape:", P1.shape)
    print("P1_corrected Shape:", P1_corrected.shape)
    spectro_compare(P1, P1_corrected, freqs_0,args.plot_output)


if __name__ == "__main__":
    main()

