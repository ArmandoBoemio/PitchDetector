import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy.signal import find_peaks
from scipy import signal


# load audio and sampling frequency from the wave file
audio, Fs = librosa.load('./audio_samples/travel.wav')

Ts = 1/Fs   # sampling period
N = len(audio)  # number of samples 
t = N / Fs  # signal duration in seconds

print("Sampling frequency:", Fs, "Hz")
print("Duration in samples", N, "samples")
print("Duration in seconds =", t, "s")


auto = sm.tsa.acf(audio, nlags=2000)
peaks = find_peaks(auto)[0] # Find peaks of the autocorrelation
lag = peaks[0] # Choose the first peak as our pitch component lag
pitch = Fs / lag # Transform lag into frequency
print('pitch =', pitch)

# FFT
X = np.fft.fft(audio)
X = X[0:int(np.ceil(N/2))]
X = np.abs(np.power(X,2))

# Spectrogram
f, t, Zxx = signal.stft(audio, Fs, nperseg=1000)


f_axis = Fs * np.arange((N/2)) / N; # frequencies axis
max_freq = 1000     # max frequency of interest
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
plt.title('Power Spectrum')
plt.plot(f_axis[0:max_freq_samples], X[0:max_freq_samples], linewidth=1)
plt.ylabel('Amplitude')
plt.xlabel('Frequency [Hz]')
plt.draw()

plt.show()
