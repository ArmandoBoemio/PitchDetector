import os
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import find_peaks
from scipy import signal


# load audio and sampling frequency from the wave file
audio, Fs = librosa.load('./tuningfork.wav')
#audio, Fs = librosa.load('./shes-a-crazy-psycho-2.wav')

Ts = 1/Fs   # sampling period
N = len(audio)  # number of samples 
t = N / Fs  # signal duration in seconds

win_len = 1000e-3
win_overlap = 0.5
win_len_samples = int(np.floor(win_len * Fs))
win_overlap_samples = int(np.floor(win_len_samples * win_overlap))
win = signal.windows.hann(win_len_samples)
#n_win = int(np.floor((len(audio)-win_overlap_samples) / (win_len_samples * win_overlap)))
n_win = int(np.floor(N / win_len_samples))


print("Sampling frequency:", Fs, "Hz")
print("Duration in samples", N, "samples")
print("Duration in seconds =", t, "s")
print("Number of windows =", n_win)
print("length of windows =", win_len_samples)



for i in range(n_win):
    #x = np.multiply(win, audio[i*win_len_samples:(i+1)*win_len_samples])
    x = audio[i*win_len_samples:(i+1)*win_len_samples]
    acf = sm.tsa.acf(x)

    # Find peaks of the autocorrelation
    peaks = find_peaks(acf)[0]
    if peaks.size:
        lag = peaks[0] # set the first peak as our pitch component lag
        pitch = Fs / lag 
        print(pitch)
    else:
        print("cant track pitch")


    #y = np.cos(2 * np.pi * pitch * np.arange(0, win_len) / Fs)



"""
# Write signal as wav (float; int16 not possible)
# Note that librosa applies some normalization and clipping
fn_out = os.path.join('audio','sine.wav')
sf.write('sine.wav', x, Fs)
"""



# FFT
X = np.fft.fft(audio)
X = X[0:int(np.ceil(N/2))]
X = np.abs(np.power(X,2))

f, t, Zxx = signal.stft(audio, Fs, nperseg=1000)

f_axis = Fs * np.arange((N/2)) / N; # frequencies axis

max_freq = 5000     # max frequency of interest
max_freq_samples = int(max_freq / (Fs) * N) # max frequency in samples


print("Plotting spectrogram...")
plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, shading='gouraud')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.draw()


# plot of the power spectrum 
print("Plotting spectrum in the range 0 to", max_freq, "Hz")
fig,ax = plt.subplots()
plt.plot(f_axis[0:max_freq_samples], X[0:max_freq_samples], linewidth=1)
plt.ylabel('Amplitude')
plt.xlabel('Frequency [Hz]')
plt.draw()

plt.show()
