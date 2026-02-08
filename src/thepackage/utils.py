from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits as pf
from numpy import log10
from scipy.optimize import curve_fit
from scipy import signal
from pathlib import Path

def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def rms(x):
    """
    Compute root-mean-square of a 1D array.

    Parameters
    ----------
    x : array-like
        Input signal

    Returns
    -------
    float
        RMS value
    """
    return (sum(v*v for v in x) / len(x)) ** 0.5

def plot_signal(x, y):
    plt.plot(x, y, label='Data')
    plt.title('Plot of the signal samples')
    plt.xlabel('Time (steps)')
    plt.ylabel('Voltage (mV)')
    plt.legend()


def minus(a,b):
    return a-b

def make_sine(fs, N, f0, A=1.0, phase=0.0):
    t = np.arange(N) / fs
    x = A * np.sin(2*np.pi*f0*t + phase)
    return t, x

def power_spectrum_fft(x, fs):
    N = len(x)
    X = np.fft.fft(x)
    f = np.fft.fftfreq(N, d=1/fs)
    P = np.abs(X)**2
    return f, P

#Oversampled spectrum by zero padding
def power_spectrum_zeropad(x, fs, Nfreq):
    N = len(x)
    X = np.fft.fft(x, n=Nfreq)             # zero padding happens automatically
    f = np.fft.fftfreq(Nfreq, d=1/fs)
    P = np.abs(X)**2
    return f, P

#shows only positive frequencies and uses log y-axis
def plot_psd_posfreq(f, P, title):
    mask = f >= 0
    plt.figure()
    plt.semilogy(f[mask], P[mask])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("|FFT|^2 (arb.)")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.show()

def load_block(npz_path, key="data"):
    with np.load(npz_path) as f:
        x = f[key].astype(float)
    return x

def power_spectrum(x, fs, remove_mean=True, window="hann"):
    x = x - np.mean(x) if remove_mean else x
    N = len(x)

    if window == "hann":
        w = np.hanning(N)
    elif window is None:
        w = np.ones(N)
    else:
        raise ValueError("window must be 'hann' or None")

    xw = x * w

    X = np.fft.fft(xw)
    f = np.fft.fftfreq(N, d=1/fs)

    # raw power (arb units); window normalization omitted for simplicity
    P = np.abs(X)**2

    # shift for plotting
    f = np.fft.fftshift(f)
    P = np.fft.fftshift(P)
    return f, P

def average_spectrum(files, fs):
    Ps = []
    f_ref = None
    for p in files:
        x = load_block(p)
        f, P = power_spectrum(x, fs)
        if f_ref is None:
            f_ref = f
        Ps.append(P)
    Pavg = np.mean(Ps, axis=0)
    return f_ref, np.array(Ps), Pavg

def average_spectrum_plot(N):
    files = []
    for i in np.arange(N):
        fs = list(data_sets.keys())[i]
        files = files+data_sets[fs]
    
    f, Pall, Pavg = average_spectrum(files, 1e6)


    mask = f >= 0
    plt.figure()
    plt.semilogy(f[mask], Pavg[mask])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (arb.)")
    plt.title(f"Average power spectrum over {len(files)} blocks (v_s={1e6/1e6:.2f} MHz)")
    plt.grid(True, which="both", alpha=0.3)
    plt.show()

def load_block(npz_path, key="data"):
    with np.load(npz_path) as f:
        x = f[key].astype(float)
    return x

def block_stats(x):
    mu = np.mean(x)
    var = np.var(x)
    sigma = np.std(x)
    rms = np.sqrt(np.mean(x**2))  # RMS voltage
    return mu, var, sigma, rms

def plot_hist_with_gaussian(x, title="", bins=80, remove_mean=True):
    x_use = x - np.mean(x) if remove_mean else x

    rms = np.sqrt(np.mean(x_use**2))  # width requested
    mu = 0.0 if remove_mean else np.mean(x_use)

    counts, edges = np.histogram(x_use, bins=bins, density=True)
    centers = 0.5*(edges[:-1] + edges[1:])

    pdf = norm.pdf(centers, loc=mu, scale=rms)

    plt.figure()
    plt.plot(centers, counts, label="Histogram (density)")
    plt.plot(centers, pdf, label=f"Gaussian (σ = RMS = {rms:.3g})")
    plt.xlabel("Voltage (arb. units)")
    plt.ylabel("Probability density")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def concat_blocks(files, Ntarget=16000):
    xs = []
    for p in files:
        xs.append(load_block(p))
        if sum(len(x) for x in xs) >= Ntarget:
            break
    x = np.concatenate(xs)
    return x[:Ntarget]

def acf_full(x):
    N = len(x)
    ac = np.correlate(x, x, mode="full") / N
    lags = np.arange(-N+1, N)
    return lags, ac

def fwhm(x, y):
    y = np.asarray(y)
    half = 0.5*np.max(y)
    idx = np.where(y >= half)[0]
    if len(idx) < 2:
        return np.nan, None, half
    return x[idx[-1]] - x[idx[0]], idx, half

def fwhm_num(x, y):
    y = np.asarray(y)
    half = 0.5 * np.max(y)
    idx = np.where(y >= half)[0]
    if len(idx) < 2:
        return np.nan
    return x[idx[-1]] - x[idx[0]]













    