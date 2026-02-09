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

def top_peaks(f, P, k=6, fmin=1e3):
    mask = (f > fmin)  # positive freq, away from DC
    ff = f[mask]
    PP = P[mask]
    idx = np.argsort(PP)[-k:][::-1]
    return list(zip(ff[idx], PP[idx]))

def plot_ps(f, P, title, fmax=None):
    m = f >= 0
    if fmax is not None:
        m = m & (f <= fmax)
    plt.figure()
    plt.semilogy(f[m], P[m])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (arb.)")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)

def mixer_filter_keep_diff(x, fs, f_diff, f_sum, bw=3e3):
    """
    Fourier-filter: zero out the SUM tone (±f_sum) and keep the difference tone.
    bw sets the half-width of the notch around f_sum.
    """
    N = len(x)
    X = np.fft.fft(x)
    f = np.fft.fftfreq(N, d=1/fs)

    notch_sum = (np.abs(f - f_sum) < bw) | (np.abs(f + f_sum) < bw)
    X_filt = X.copy()
    X_filt[notch_sum] = 0.0

    x_filt = np.fft.ifft(X_filt).real
    return x_filt

def signal_data(npz_name):
        npzfile = np.load(f'{npz_name}.npz')
        plt.plot(np.arange(len(npzfile['data'])), np.fft.fftshift(npzfile['data']), c='g')
        #plt.xlim(0, 1000)
        #plt.ylim(-30, 30)
        plt.xlabel('Sample Number')
        plt.ylabel('Amplitude (arb.)')
        plt.title('Frequency Resolution Raw Data')
        plt.show()

def voltage_plot(npz_name):
        npzfile = np.load(f'{npz_name}.npz')
        vspec = np.fft.fft(npzfile['data'])
        plt.plot(np.arange(len(vspec))-1000, np.real(vspec), label='Real Component', c='r')
        plt.plot(np.arange(len(vspec))-1000, np.imag(vspec), label='Imaginary Component', c='b')
        plt.xlabel('Frequency (arb.)')
        plt.ylabel('Voltage (arb.)')
        plt.xlim(-1000, 1000)
        plt.legend()
        plt.title('Real versus Imaginary components of the Voltage Spectra')
        plt.show()
    
def voltage_plot_detail(npz_name, x_lower, x_upper):
        npzfile = np.load(f'{npz_name}.npz')
        vspec = np.fft.fft(npzfile['data'])
        plt.plot(np.arange(len(vspec))-1000, np.real(vspec), label='Real Component', c='r')
        plt.plot(np.arange(len(vspec))-1000, np.imag(vspec), label='Imaginary Component', c='b')
        plt.xlabel('Frequency (arb.)')
        plt.ylabel('Voltage (arb.)')
        plt.xlim(x_lower, x_upper)
        plt.ylim(-2000,2000)
        plt.legend()
        plt.title('Real versus Imaginary components of the Voltage Spectra (Zoomed in)')
        plt.show()

def spectrum_other_nyquist_windows(file, fs, W=4, fmax=None, title=None):
    """
    Build power spectrum over ±W*fs/2 by tiling the baseband FFT (periodic with fs).
    W>=4 recommended by manual.
    """
    x = load_block(file)
    f, P = power_spectrum(x, fs)     # your function returns fftshifted f in [-fs/2, fs/2)

    # choose which Nyquist windows to include: k = ...,-2,-1,0,1,2,... (W total)
    kmin = -W//2
    kmax =  W//2
    ks = range(kmin, kmax)  # length W

    # tile spectrum by shifting frequency axis by k*fs
    f_ext = np.concatenate([f + k*fs for k in ks])
    P_ext = np.concatenate([P for _ in ks])

    # sort for a clean line plot
    idx = np.argsort(f_ext)
    f_ext, P_ext = f_ext[idx], P_ext[idx]

    # optional crop for readability
    if fmax is None:
        fmax = (W*fs/2)
    m = np.abs(f_ext) <= fmax

    plt.figure(figsize=(8,4))
    plt.semilogy(f_ext[m], np.maximum(P_ext[m], 1e-30))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (arb.)")
    plt.title(title or f"Power spectrum across Nyquist windows (W={W}, fs={fs/1e6:.2f} MHz)")
    plt.grid(True, alpha=0.3)
    plt.show()

    return f_ext, P_ext


def compare_windows_numbers_only(file, fs, W=4, n_peaks=1, guard_hz=1e3):
    """
    Quick numeric check: find the strongest peak frequency in each Nyquist window.
    Useful for 'compare across windows' without extra plots.
    """
    x = load_block(file)
    f, P = power_spectrum(x, fs)

    ks = range(-W//2, W//2)
    out = []
    for k in ks:
        f_k = f + k*fs
        P_k = P.copy()

        # ignore a tiny region near 0 Hz *in that window's coordinates* if you want
        m = np.abs(f) > guard_hz
        fpk = f_k[m][np.argmax(P_k[m])]
        out.append((k, fpk))

    print("Strongest peak per Nyquist window (k, f_peak_Hz):")
    for k, fpk in out:
        print(f"  k={k:>2d}  f_peak={fpk: .3f} Hz")
    return out

def iq_phase_at_tone(I_path, Q_path, fs):
    I = load_block(I_path) - np.mean(load_block(I_path))
    Q = load_block(Q_path) - np.mean(load_block(Q_path))
    N = len(I)

    FI = np.fft.rfft(I)
    FQ = np.fft.rfft(Q)
    f  = np.fft.rfftfreq(N, d=1/fs)

    # dominant tone (ignore DC)
    k = np.argmax(np.abs(FI[1:])**2 + np.abs(FQ[1:])**2) + 1

    phi_I = np.angle(FI[k])
    phi_Q = np.angle(FQ[k])
    dphi  = np.angle(np.exp(1j*(phi_Q - phi_I)))  # wrap to [-pi, pi]

    return f[k], np.degrees(dphi)

def as_1d(x, col=0):
    """
    Convert loaded SDR block to a 1-D vector.
    If x is (N,2), pick one column (default 0).
    If x is already (N,), return as-is.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        return x
    if x.ndim == 2:
        return x[:, col]
    raise ValueError(f"Unexpected data shape: {x.shape}")

def analyze_internal_mixer(I_path, Q_path, fs, title="", fmin=1e3, fmax=None, nshow=2000, col=0):
    I = as_1d(load_block(I_path), col=col)
    Q = as_1d(load_block(Q_path), col=col)

    N = min(len(I), len(Q))
    I, Q = I[:N], Q[:N]

    z = (I - np.mean(I)) + 1j*(Q - np.mean(Q))

    f, P = power_spectrum(z, fs)

    m = (np.abs(f) > fmin)
    if fmax is not None:
        m &= (np.abs(f) < fmax)

    f_peak = f[m][np.argmax(P[m])]
    sign = "+Δν (upper sideband)" if f_peak > 0 else "−Δν (lower sideband)"
    print(f"{title}\nPeak at {f_peak:.2f} Hz → {sign}\n")

    plt.figure()
    plt.semilogy(f, np.maximum(P, 1e-30))
    if fmax is not None:
        plt.xlim(-fmax, fmax)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (arb.)")
    plt.title(f"{title} power spectrum (I + jQ)")
    plt.grid(True, alpha=0.3)
    plt.show()

    t = np.arange(N) / fs
    plt.figure()
    plt.plot(t[:nshow]*1e3, np.real(z[:nshow]), label="I (real)")
    plt.plot(t[:nshow]*1e3, np.imag(z[:nshow]), label="Q (imag)", alpha=0.8)
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (arb.)")
    plt.title(f"{title} waveform (I and Q)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    return f, P, f_peak





    