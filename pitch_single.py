import librosa
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import scipy as sc
from scipy.signal import butter, lfilter, freqz, find_peaks, peak_prominences


# filter functions
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a
    
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y



# load audio and sampling frequency
#audio, Fs = librosa.load('./sine65.wav')
#audio, Fs = librosa.load('./shes-a-crazy-psycho-2.wav')
audio, Fs = librosa.load('./tuningfork.wav')


T = 1/Fs # Sampling period
N = len(audio) # Signal length in samples
t = N / Fs # Signal length in seconds

print("Sampling frequency:", Fs, "Hz")
print("Duration in samples", N, "samples")
print("Duration in seconds =", t, "s")

# low-pass
Fc = 500
filt_ord = 6
audio_lp = butter_lowpass_filter(audio, Fc, Fs, filt_ord)

# spectrums
X = np.fft.fft(audio)[0:int(N/2)]/N 
X[1:] = 2*X[1:]
P = np.abs(X) # power spectrum

X_lp = np.fft.fft(audio_lp)[0:int(N/2)]/N 
X_lp[1:] = 2*X_lp[1:]
P_lp = np.abs(X_lp) # power spectrum of the low-passed audio


# autocorrelation
acf = sm.tsa.acf(audio_lp, nlags=5000)

peaks = find_peaks(acf)[0] # Find peaks of the autocorrelation
print("peak found =", peaks)
prominences = peak_prominences(acf, peaks)[0]
#print("prominences found =", prominences)
max_prom = np.argmax(prominences)

#lag = peaks[max_prom] # Choose the most prominent peak as our pitch component lag
lag = peaks[0] # Choose the first peak as our pitch component lag

pitch = Fs / lag # Transform lag into frequency
print ('pitch = ', pitch, 'Hz')

# plotting
f_axis = Fs * np.arange((N/2)) / N; # frequencies

fig, (ax1, ax2) = plt.subplots(2,1)
ax1.plot(f_axis[0:10000], P[0:10000], linewidth=2)
ax2.plot(f_axis[0:10000], P_lp[0:10000], linewidth=2)
plt.ylabel('Amplitude')
plt.xlabel('Frequency [Hz]')
plt.show()
#sm.graphics.tsa.plot_acf(acf)
#plt.show()