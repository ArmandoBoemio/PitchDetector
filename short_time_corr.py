import librosa
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from scipy.signal import butter, lfilter, correlate, correlation_lags

# filter functions
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='highpass', analog=False)
    return b, a    
    
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# non linear functions: clipping and compress
def clc(x, cl):
    y = np.zeros(len(x))
    x_mean = np.mean(x)
    x = x - x_mean
    for i in range(len(x)):
        if x[i] <= -cl:
            y[i] = x[i] + cl
        elif x[i] >= cl:
            y[i] = x[i] - cl
        else:
            y[i] = 0
    return y + x_mean

def main():
    # load audio and sampling frequency
    audio, Fs = librosa.load('./audio_samples/female_voice_1.wav')

    # trim silences at start and end
    audio, _ = librosa.effects.trim(audio, top_db=60)


    F_min = 40
    F_max = 600

    acf_min = int(round(Fs / F_max))
    acf_max = int(round(Fs / F_min))

    T = 1/Fs # Sampling period
    N = len(audio) # Signal length in samples
    t = N / Fs # Signal length in seconds

    print("Sampling frequency:", Fs, "Hz")
    print("Duration in samples", N, "samples")
    print("Duration in seconds =", t, "s")


    # low-pass
    Fc_low = 600 # can be set up to 900 Hz
    filt_ord_low = 6
    audio_filt = butter_lowpass_filter(audio, Fc_low, Fs, filt_ord_low)

    # high-pass - for virtual pitch testing
    #Fc_hi = 300
    #filt_ord_hi = 4
    #audio_filt = butter_highpass_filter(audio, Fc_hi, Fs, filt_ord_hi)


    # listen to the original audio
    sd.play(audio, Fs)
    #sd.wait() # used to block the execution of the script while listening the audio

    # short time analysis parameters
    frame_len = 30e-3 # generally, must be set higher for lower pitches
    N_a = int(np.floor(frame_len * Fs)) 
    n_win = int(np.floor(N / N_a)) # no overlap

    fixed_len = N_a * n_win
    audio_filt = audio_filt[0:fixed_len]

    print("Length of a window =", N_a)
    print("Number of windows =", n_win)


    # short-time analysis
    pitch = np.zeros(n_win)
    audio_slices = np.zeros((N_a, n_win))

    for i in range(n_win):
        audio_slices[:,i] = audio_filt[i*N_a:(i+1)*N_a]

    max_values = np.amax(audio_slices, axis=0)
    n_cl = round(n_win/3)
    cl = 0.68 * min(max_values[:n_cl]+max_values[-n_cl:]) # clipping threshold value

    x_clc = clc(audio_filt, cl)
    x_clp = np.clip(audio_filt, -cl, cl)

    for i in range(n_win):
        slice_clc = x_clc[i*N_a:(i+1)*N_a]
        slice_clp = x_clp[i*N_a:(i+1)*N_a]

        x_corr = correlate(slice_clc, slice_clp, mode='full')
        x_corr = x_corr[len(x_corr)//2:] #clip to only positive values
        x_corr = x_corr[acf_min:acf_max]
        
        lags = correlation_lags(slice_clc.size, slice_clp.size, mode="full")
        lags = lags[len(lags)//2:]
        lags = lags[acf_min:acf_max]
        lag = lags[np.argmax(x_corr)]
        
        pitch[i] = Fs / np.abs(lag)
        # print('pitch =', pitch[i]) 
        
    '''#plot of data for each window
        fig, (ax_orig, ax_clc, ax_clp, ax_corr) = plt.subplots(4, 1, figsize=(4.8, 4.8))
        ax_orig.plot(audio_slices[:,i])
        ax_orig.set_title('Low pass')
        ax_orig.set_xlabel('Sample Number')
        ax_clp.plot(slice_clp)
        ax_clp.set_title('Clip')
        ax_clp.set_xlabel('Sample Number')
        ax_clc.plot(slice_clc)
        ax_clc.set_title('Clip and compress')
        ax_clc.set_xlabel('Sample Number')
        ax_corr.plot(lags, x_corr)
        ax_corr.set_title('Cross-correlated signal')
        ax_corr.set_xlabel('Lag')
        ax_orig.margins(0, 0.1)
        ax_clc.margins(0, 0.1)
        ax_clp.margins(0, 0.1)
        ax_corr.margins(0, 0.1)
        fig.tight_layout()
        plt.show()}
        '''




    yin = librosa.yin(audio, fmin=F_min, fmax=F_max, sr=Fs, frame_length=N_a, hop_length=N_a)

    t_axis = np.arange(len(pitch))
    yin_axis = np.arange(len(yin))

    fig, (ax, ax_yin) = plt.subplots(2,1, figsize=(4.8, 4.8))
    ax.plot(t_axis, pitch, linewidth=2)
    ax.set(ylim=(min(pitch), max(pitch)))
    ax.margins(0, 35)
    ax.set_title('Low pass + NL')
    ax.set_xlabel('Window Index')
    ax.set_ylabel('Pitch [Hz]')

    ax_yin.plot(yin_axis, yin, linewidth=2)
    ax_yin.set(ylim=(min(yin), max(yin)))
    ax_yin.margins(0, 35)
    ax_yin.set_title('YIN')
    ax_yin.set_ylabel('Pitch [Hz]')
    ax_yin.set_xlabel('Window index')
    fig.tight_layout()

    plt.draw()
    plt.show()

if __name__ == '__main__':
    main()

