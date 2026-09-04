import socket
import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import windows
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os
import argparse

# argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Briggs Algorithm")
    parser.add_argument("input_file", help="Input file containing noisy sine data")
    parser.add_argument("--FFT-size", type=int, default=2048, help="Size of the FFT window")
    parser.add_argument("--overlap", type=int, default=1024, help="Overlap between FFT windows")
    parser.add_argument("--plot-output", type=str, default=None, help="Output file to save the comparison plot")
    parser.add_argument("--threshold-factor", type=float, default=6.0, help="Threshold factor for RFI detection (in units of MAD)")
    args = parser.parse_args()
    if args.plot_output is None:
        args.plot_output = 'comparison_spectrogram.png'
    return args

def briggs_rfi_cancellation(ch1, ch2, ch3, ch4, fs=61440000, nfft=2048, overlap=1024, threshold_factor=6):
    """
    Implements F.H. Briggs RFI post-correlation cancellation keeping time history 
    intact for spectrogram visualization.
    """
    # Cast raw integer ADC counts to floats for FFT precision
    v1_raw = ch1.astype(np.float32)
    v2_raw = ch2.astype(np.float32)
    v3_raw = ch3.astype(np.float32)
    v4_raw = ch4.astype(np.float32)

    hop = nfft - overlap
    n_segments = (len(ch1) - nfft) // hop + 1
    n_freqs = nfft // 2 + 1
    
    # Initialize 2D arrays to retain the time (segment) axis
    P11 = np.zeros((n_segments, n_freqs))
    P22 = np.zeros((n_segments, n_freqs))
    P13 = np.zeros((n_segments, n_freqs), dtype=complex)
    P14 = np.zeros((n_segments, n_freqs), dtype=complex)
    P23 = np.zeros((n_segments, n_freqs), dtype=complex)
    P24 = np.zeros((n_segments, n_freqs), dtype=complex)
    P34 = np.zeros((n_segments, n_freqs), dtype=complex)
    
    window = np.hanning(nfft)
    
    # Extract short-time windows
    for i in range(n_segments):
        start = i * hop
        end = start + nfft
        
        # Apply window and FFT (one-sided)
        v1 = np.fft.rfft(v1_raw[start:end] * window)
        v2 = np.fft.rfft(v2_raw[start:end] * window)
        v3 = np.fft.rfft(v3_raw[start:end] * window)
        v4 = np.fft.rfft(v4_raw[start:end] * window)
        
        # Store individual time frame powers
        P11[i, :] = np.abs(v1)**2
        P22[i, :] = np.abs(v2)**2
        
        P13[i, :] = v1 * np.conj(v3)
        P14[i, :] = v1 * np.conj(v4)
        P23[i, :] = v2 * np.conj(v3)
        P24[i, :] = v2 * np.conj(v4)
        P34[i, :] = v3 * np.conj(v4)
        
        # Calculate closure-based RFI power models per time frame
    eps = 1e-18
    M1 = (P13 * np.conj(P14)) / (np.conj(P34) + eps)
    M2 = (P23 * np.conj(P24)) / (np.conj(P34) + eps)
    
    
    # Take the real part of the models
    M1_real = np.real(M1)
    M2_real = np.real(M2)
    
    #Compute the magnitude of the reference cross-correlation
    P34_mag = np.abs(P34)
    
    #Calculate the median and MAD across the entire time-frequency plane
    median_val = np.median(P34_mag)
    mad_val = np.median(np.abs(P34_mag - median_val))
    
    # Establish a standard threshold (typically 5 to 7 sigma)
    # 1.4826 scales the MAD to equal a standard deviation for a normal distribution
    sigma_estimate = 1.4826 * mad_val
    threshold = median_val + (threshold_factor * sigma_estimate)
    
    #Create a boolean mask where true, coherent RFI exceeds the noise threshold
    rfi_mask = P34_mag > threshold
    
    #Apply the mask: keep the model where RFI exists, set to 0.0 everywhere else
    M1_cleaned_model = np.where(rfi_mask, M1_real, 0.0)
    M2_cleaned_model = np.where(rfi_mask, M2_real, 0.0)
    
    #Perform the targeted subtraction
    P11_clean = P11 - M1_cleaned_model
    P22_clean = P22 - M2_cleaned_model
    
    # Enforce a strict minimum floor to guarantee no negative power values 
    # slip into log plots
    P11_clean = np.maximum(P11_clean, 1e-10)
    P22_clean = np.maximum(P22_clean, 1e-10)
    
    freqs = np.fft.rfftfreq(nfft, d=1/fs)
    return freqs, P11, P11_clean, P22_clean, M1, M2


def spectro_compare(P1, P1_corrected, plot_output):
    # Enforce threshold to avoid taking log of zero or negatives
    P1_dB = 10 * np.log10(np.clip(P1, 1e-12, None))
    P1_corrected_dB = 10 * np.log10(np.clip(P1_corrected, 1e-12, None))
    
    # Find common scale limits across both plots
    vmin = np.min([P1_dB, P1_corrected_dB])
    vmax = np.max([P1_dB, P1_corrected_dB])

    # Plot spectrogram comparison panels
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=True)
    
    # Transposing (.T) maps frequency to Y-axis and windowed blocks to X-axis
    im1 = ax1.imshow(P1_dB.T, aspect='auto', origin='lower', vmin=vmin, vmax=vmax, cmap='viridis')
    ax1.set_ylabel('Frequency Channel')
    ax1.set_title('Original Power Spectral Density (with RFI Tones)')
    plt.colorbar(im1, ax=ax1, label='Power (dB)')
    
    im2 = ax2.imshow(P1_corrected_dB.T, aspect='auto', origin='lower', vmin=vmin, vmax=vmax, cmap='viridis')
    ax2.set_ylabel('Frequency Channel')
    ax2.set_xlabel('Time Window Index')
    ax2.set_title('Corrected Power Spectral Density (Briggs Subtracted)')
    plt.colorbar(im2, ax=ax2, label='Power (dB)')

    plt.tight_layout()
    if plot_output:
        plt.savefig(plot_output)
    plt.show()

def model_compare(M1, M2, plot_output):
    # Enforce threshold to avoid taking log of zero or negatives
    M1_dB = 10 * np.log10(np.clip(np.real(M1), 1e-12, None))
    M2_dB = 10 * np.log10(np.clip(np.real(M2), 1e-12, None))
    
    # Find common scale limits across both plots
    vmin = np.min([M1_dB, M2_dB])
    vmax = np.max([M1_dB, M2_dB])

    # Plot model comparison panels
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=True)
    
    im1 = ax1.imshow(M1_dB.T, aspect='auto', origin='lower', vmin=vmin, vmax=vmax, cmap='plasma')
    ax1.set_ylabel('Frequency Channel')
    ax1.set_title('RFI Model for Channel 1 (M1)')
    plt.colorbar(im1, ax=ax1, label='Power (dB)')
    
    im2 = ax2.imshow(M2_dB.T, aspect='auto', origin='lower', vmin=vmin, vmax=vmax, cmap='plasma')
    ax2.set_ylabel('Frequency Channel')
    ax2.set_xlabel('Time Window Index')
    ax2.set_title('RFI Model for Channel 2 (M2)')
    plt.colorbar(im2, ax=ax2, label='Power (dB)')

    plt.tight_layout()
    if plot_output:
        model_name = os.path.splitext(plot_output)[0] + '_models.png'
        plt.savefig(model_name)
    plt.show()

def main():
    args = parse_args()
    Fsamp = 61440000
    
    print(f"Loading CSV data from: {args.input_file}")
    # Unpack the 4 comma-separated integer streams
    dat0, dat1, dat2, dat3 = np.genfromtxt(args.input_file, delimiter=',', dtype=np.int16, unpack=True)

    # Compute Briggs matrix transformations
    freqs, P11_orig, P11_clean, P22_clean, M1, M2 = briggs_rfi_cancellation(
        dat0, dat1, dat2, dat3, 
        fs=Fsamp, 
        nfft=args.FFT_size, 
        overlap=args.overlap
    )

    print("Original Matrix Shape:", P11_orig.shape)
    print("Cleaned Matrix Shape:", P11_clean.shape)

    # plot RFI power models as spectrograms for visual comparison
    model_compare(M1, M2, args.plot_output)
    
    # Provide the original and corrected matrices for Channel 1 to the layout function
    spectro_compare(P11_orig, P11_clean, args.plot_output)

    #save the cleaned matrix to CSV for further analysis
    clean_name = os.path.splitext(args.plot_output)[0] + '_P11_clean.csv'
    orig_name = os.path.splitext(args.plot_output)[0] + '_P11_orig.csv'
    np.savetxt(clean_name, P11_clean, delimiter=',', fmt='%.6e')
    np.savetxt(orig_name, P11_orig, delimiter=',', fmt='%.6e')

if __name__ == "__main__":
    main()
