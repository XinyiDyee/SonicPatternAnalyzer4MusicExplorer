import numpy as np
import os, sys, librosa
from scipy import signal
from scipy.interpolate import interp1d
from scipy import ndimage
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import IPython.display as ipd
import pandas as pd

def plot_wav_spectrogram(fn_wav, xlim=None, audio=True):
    """
    Plot waveform and computed spectrogram of an audio file and optionally play the audio.
    
    Parameters:
    fn_wav (str): Path to the audio file
    xlim (tuple, optional): Time range to display and play (in seconds)
    audio (bool, optional): Whether to return an audio object
    """
    # Load the audio file
    y, sr = librosa.load(fn_wav)
    
    # If xlim is specified, extract that segment of audio
    if xlim is not None:
        start_sample = int(xlim[0] * sr)
        end_sample = int(xlim[1] * sr)
        y = y[start_sample:end_sample]
    
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 8))
    
    # Plot the waveform
    librosa.display.waveshow(y, sr=sr, ax=ax1)
    ax1.set_title('Waveform')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Amplitude')
    
    # Compute and plot the spectrogram
    D = librosa.stft(y)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='hz', sr=sr, ax=ax2)
    ax2.set_title('Spectrogram')
    ax2.set_xlabel('Time (seconds)')
    fig.colorbar(img, ax=ax2, format="%+2.0f dB")
    
    # Set x-axis limits and ticks
    if xlim is not None:
        ax1.set_xlim(0, xlim[1] - xlim[0])
        ax2.set_xlim(0, xlim[1] - xlim[0])
    else:
        ax1.set_xlim(0, duration)
        ax2.set_xlim(0, duration)
    
    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()
    
    # Return audio object if requested
    if audio:
        return ipd.Audio(y, rate=sr)
    

def read_annotation_pos(fn_ann, label='', header=True, print_table=False):
    """
    Read beat annotation from a CSV file.

    Args:
        fn_ann (str): Name of the CSV file containing beat annotations
        print_table (bool): Prints table if True (Default value = False)

    Returns:
        ann (list): List of annotations
        label_keys (dict): Dictionary specifying color and line style used for labels
    """

    # Read CSV file
    df = pd.read_csv(fn_ann, sep='\t', header=None, names=['time', 'beat'])
    
    if print_table:
        print(df)
    
    # # Convert DataFrame to list of lists
    # ann = df.values.tolist()

    # label_keys = {'beat': {'linewidth': 2, 'linestyle': ':', 'color': 'r'},
    #               'onset': {'linewidth': 1, 'linestyle': ':', 'color': 'r'}}
    
    return df



def plot_signal(x, Fs=1, T_coef=None, ax=None, figsize=(6, 2), xlabel='Time (seconds)', ylabel='', title='', dpi=72,
                ylim=True, **kwargs):
    """Line plot visualization of a signal, e.g. a waveform or a novelty function.

    Args:
        x: Input signal
        Fs: Sample rate (Default value = 1)
        T_coef: Time coeffients. If None, will be computed, based on Fs. (Default value = None)
        ax: The Axes instance to plot on. If None, will create a figure and axes. (Default value = None)
        figsize: Width, height in inches (Default value = (6, 2))
        xlabel: Label for x axis (Default value = 'Time (seconds)')
        ylabel: Label for y axis (Default value = '')
        title: Title for plot (Default value = '')
        dpi: Dots per inch (Default value = 72)
        ylim: True or False (auto adjust ylim or nnot) or tuple with actual ylim (Default value = True)
        **kwargs: Keyword arguments for matplotlib.pyplot.plot

    Returns:
        fig: The created matplotlib figure or None if ax was given.
        ax: The used axes.
        line: The line plot
    """
    fig = None
    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.subplot(1, 1, 1)
    if T_coef is None:
        T_coef = np.arange(x.shape[0]) / Fs

    if 'color' not in kwargs:
        kwargs['color'] = 'gray'

    line = ax.plot(T_coef, x, **kwargs)

    ax.set_xlim([T_coef[0], T_coef[-1]])
    if ylim is True:
        ylim_x = x[np.isfinite(x)]
        x_min, x_max = ylim_x.min(), ylim_x.max()
        if x_max == x_min:
            x_max = x_max + 1
        ax.set_ylim([min(1.1 * x_min, 0.9 * x_min), max(1.1 * x_max, 0.9 * x_max)])
    elif ylim not in [True, False, None]:
        ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if fig is not None:
        plt.tight_layout()

    return fig, ax, line



def compute_novelty_function(y, sr, hop_length=512):
    """
    Compute novelty function from audio signal.
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)
    return onset_env, times



def plot_novelty_and_beats(fn_wav, fn_ann, xlim=None, audio=True):
    """
    Plot novelty function and beat annotations, and optionally play the audio.
    
    Parameters:
    fn_wav (str): Path to the audio file
    fn_ann (str): Path to the text file containing beat annotations
    xlim (tuple, optional): Time range to display and play (in seconds)
    audio (bool, optional): Whether to return an audio object
    """
    # Load the audio file
    y, sr = librosa.load(fn_wav)
    
    # Compute novelty function
    novelty, nov_times = compute_novelty_function(y, sr)
    
    # Read beat annotations
    ann = read_annotation_pos(fn_ann).values.tolist()
    
    # If xlim is specified, extract that segment of audio and filter annotations
    if xlim is not None:
        start_sample = int(xlim[0] * sr)
        end_sample = int(xlim[1] * sr)
        y = y[start_sample:end_sample]
        novelty = novelty[int(xlim[0] * sr / 512):int(xlim[1] * sr / 512)]
        nov_times = nov_times[int(xlim[0] * sr / 512):int(xlim[1] * sr / 512)] - xlim[0]
        ann = [a for a in ann if xlim[0] <= a[0] < xlim[1]]
        for a in ann:
            a[0] -= xlim[0]  # Adjust annotation times

    duration = librosa.get_duration(y=y, sr=sr)
    
    # Create plot
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(12, 6))
    
    # Plot novelty function
    ax[0].plot(nov_times, novelty, color='k')
    ax[0].set_title('Novelty function')
    ax[0].set_ylabel('Novelty')
    
    # Plot beat annotations
    for time, beat in ann:
        ax[1].axvline(x=time, color='r', linestyle=':', linewidth=2)
    
    ax[1].set_title('Annotated beat positions')
    ax[1].set_xlabel('Time (seconds)')
    ax[1].set_yticks([])  # Remove y-axis ticks for beat annotations
    
    # Set x-axis limits
    if xlim is not None:
        ax[0].set_xlim(0, xlim[1] - xlim[0])
        ax[1].set_xlim(0, xlim[1] - xlim[0])
    else:
        ax[0].set_xlim(0, duration)
        ax[1].set_xlim(0, duration)
    
    plt.tight_layout()
    plt.show()
    
    # Return audio object if requested
    if audio:
        return ipd.Audio(y, rate=sr)
    


def compute_local_average(x, M):
    """Compute local average of signal

    Args:
        x (np.ndarray): Signal
        M (int): Determines size (2M+1) in samples of centric window  used for local average

    Returns:
        local_average (np.ndarray): Local average signal
    """
    L = len(x)
    local_average = np.zeros(L)
    for m in range(L):
        a = max(m - M, 0)
        b = min(m + M + 1, L)
        local_average[m] = (1 / (2 * M + 1)) * np.sum(x[a:b])
    return local_average



def compute_novelty_spectrum(x, Fs=1, N=1024, H=256, gamma=100.0, M=10, norm=True):
    """Compute spectral-based novelty function

    Args:
        x (np.ndarray): Signal
        Fs (scalar): Sampling rate (Default value = 1)
        N (int): Window size (Default value = 1024)
        H (int): Hop size (Default value = 256)
        gamma (float): Parameter for logarithmic compression (Default value = 100.0)
        M (int): Size (frames) of local average (Default value = 10)
        norm (bool): Apply max norm (if norm==True) (Default value = True)

    Returns:
        novelty_spectrum (np.ndarray): Energy-based novelty function
        Fs_feature (scalar): Feature rate
    """
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hann')
    Fs_feature = Fs / H
    Y = np.log(1 + gamma * np.abs(X))
    Y_diff = np.diff(Y)
    Y_diff[Y_diff < 0] = 0
    novelty_spectrum = np.sum(Y_diff, axis=0)
    novelty_spectrum = np.concatenate((novelty_spectrum, np.array([0.0])))
    if M > 0:
        local_average = compute_local_average(novelty_spectrum, M)
        novelty_spectrum = novelty_spectrum - local_average
        novelty_spectrum[novelty_spectrum < 0] = 0.0
    if norm:
        max_value = max(novelty_spectrum)
        if max_value > 0:
            novelty_spectrum = novelty_spectrum / max_value
    return novelty_spectrum, Fs_feature


def compute_penalty(N, beat_ref):
    """| Compute penalty funtion used for beat tracking
    | Note: Concatenation of '0' because of Python indexing conventions

    Args:
        N (int): Length of vector representing penalty function
        beat_ref (int): Reference beat period (given in samples)

    Returns:
        penalty (np.ndarray): Penalty function
    """
    t = np.arange(1, N) / beat_ref
    penalty = -np.square(np.log2(t))
    t = np.concatenate((np.array([0]), t))
    penalty = np.concatenate((np.array([0]), penalty))
    return penalty


def compute_beat_sequence(novelty, beat_ref, penalty=None, factor=1.0, return_all=False):
    """| Compute beat sequence using dynamic programming 
    | Note: Concatenation of '0' because of Python indexing conventions

    Args:
        novelty (np.ndarray): Novelty function
        beat_ref (int): Reference beat period
        penalty (np.ndarray): Penalty function (Default value = None)
        factor (float): Weight parameter for adjusting the penalty (Default value = 1.0)
        return_all (bool): Return details (Default value = False)

    Returns:
        B (np.ndarray): Optimal beat sequence
        D (np.ndarray): Accumulated score
        P (np.ndarray): Maximization information
    """
    N = len(novelty)
    if penalty is None:
        penalty = compute_penalty(N, beat_ref)
    penalty = penalty * factor
    novelty = np.concatenate((np.array([0]), novelty))
    D = np.zeros(N+1)
    P = np.zeros(N+1, dtype=int)
    D[1] = novelty[1]
    P[1] = 0
    # forward calculation
    for n in range(2, N+1):
        m_indices = np.arange(1, n)
        scores = D[m_indices] + penalty[n-m_indices]
        maxium = np.max(scores)
        if maxium <= 0:
            D[n] = novelty[n]
            P[n] = 0
        else:
            D[n] = novelty[n] + maxium
            P[n] = np.argmax(scores) + 1
    # backtracking
    B = np.zeros(N, dtype=int)
    k = 0
    B[k] = np.argmax(D)
    while P[B[k]] != 0:
        k = k+1
        B[k] = P[B[k-1]]
    B = B[0:k+1]
    B = B[::-1]
    B = B - 1
    if return_all:
        return B, D, P
    else:
        return B


def beat_period_to_tempo(beat, Fs):
    """Convert beat period (samples) to tempo (BPM)

    Args:
        beat (int): Beat period (samples)
        Fs (scalar): Sample rate

    Returns:
        tempo (float): Tempo (BPM)
    """
    tempo = 60 / (beat / Fs)
    return tempo


    
def compute_plot_sonify_beat(x, Fs, nov, Fs_nov, beat_ref, factor, title=None, figsize=(6, 2)):
    """Compute, plot, and sonify beat sequence from novelty function

    Args:
        x: Novelty function
        Fs: Sample rate
        nov: Novelty function
        Fs_nov: Rate of novelty function
        beat_ref: Reference beat period
        factor: Weight parameter for adjusting the penalty
        title: Title of figure (Default value = None)
        figsize: Size of figure (Default value = (6, 2))
    """
    B = compute_beat_sequence(nov, beat_ref=beat_ref, factor=factor)

    beats = np.zeros(len(nov))
    beats[np.array(B, dtype=np.int32)] = 1
    if title is None:
        tempo = beat_period_to_tempo(beat_ref, Fs_nov)
        title = (r'Optimal beat sequence ($\hat{\delta}=%d$, $F_\mathrm{s}=%d$, '
                 r'$\hat{\tau}=%0.0f$ BPM, $\lambda=%0.2f$)' % (beat_ref, Fs_nov, tempo, factor))

    fig, ax, line = plot_signal(nov, Fs_nov, color='k', title=title, figsize=figsize)
    T_coef = np.arange(nov.shape[0]) / Fs_nov
    ax.plot(T_coef, beats, ':r', linewidth=1)
    plt.show()

    beats_sec = T_coef[B]
    x_peaks = librosa.clicks(times=beats_sec, sr=Fs, click_freq=1000, length=len(x))
    ipd.display(ipd.Audio(x + x_peaks, rate=Fs))

    return beats_sec


def resample_signal(x_in, Fs_in, Fs_out=100, norm=True, time_max_sec=None, sigma=None):
    """Resample and smooth signal

    Args:
        x_in (np.ndarray): Input signal
        Fs_in (scalar): Sampling rate of input signal
        Fs_out (scalar): Sampling rate of output signal (Default value = 100)
        norm (bool): Apply max norm (if norm==True) (Default value = True)
        time_max_sec (float): Duration of output signal (given in seconds) (Default value = None)
        sigma (float): Standard deviation for smoothing Gaussian kernel (Default value = None)

    Returns:
        x_out (np.ndarray): Output signal
        Fs_out (scalar): Feature rate of output signal
    """
    if sigma is not None:
        x_in = ndimage.gaussian_filter(x_in, sigma=sigma)
    T_coef_in = np.arange(x_in.shape[0]) / Fs_in
    time_in_max_sec = T_coef_in[-1]
    if time_max_sec is None:
        time_max_sec = time_in_max_sec
    N_out = int(np.ceil(time_max_sec*Fs_out))
    T_coef_out = np.arange(N_out) / Fs_out
    if T_coef_out[-1] > time_in_max_sec:
        x_in = np.append(x_in, [0])
        T_coef_in = np.append(T_coef_in, [T_coef_out[-1]])
    x_out = interp1d(T_coef_in, x_in, kind='linear')(T_coef_out)
    if norm:
        x_max = max(x_out)
        if x_max > 0:
            x_out = x_out / max(x_out)
    return x_out, Fs_out
