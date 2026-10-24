"""Features per 1 s window (20 kHz) from NLN-EMP-style signals.

BASE (15): 10 vibration (averaged over the 5 accelerometers), 4 current, 1 voltage.
EXTRA (6, all read off the FFT already computed): shaft speed estimated from the current fundamental,
vibration energy at 1x/2x/3x shaft speed, and energy around the BPFO/BPFI bearing fault frequencies.
Shaft speed is estimated as SLIP_RATIO * supply frequency (2-pole motor, ~2 % slip: an assumption);
BPFO/BPFI ratios are textbook values, real bearings differ.
"""
import numpy as np
from scipy import stats

FS = 20000
WINDOW = FS  # 1 s, non-overlapping
SLIP_RATIO = 0.98
BPFO_RATIO, BPFI_RATIO = 3.05, 4.95

VIB_FEATURES = ["vib_rms", "vib_peak", "vib_crest", "vib_kurtosis", "vib_skew",
                "vib_band_low", "vib_band_high", "vib_centroid", "vib_entropy", "vib_dom_freq"]
ELEC_FEATURES = ["cur_rms", "cur_unbalance", "cur_thd", "cur_sideband_ratio", "volt_unbalance"]
BASE_FEATURES = VIB_FEATURES + ELEC_FEATURES
EXTRA_FEATURES = ["shaft_hz", "vib_h1", "vib_h2", "vib_h3", "vib_bpfo_band", "vib_bpfi_band"]
FEATURE_NAMES = BASE_FEATURES + EXTRA_FEATURES  # all 21; BASE_FEATURES is the first 15


def _psd(x, fs=FS):
    x = x - x.mean()
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    return np.fft.rfftfreq(len(x), 1 / fs), p


def _vib_channel(x, fr, extended):
    f, p = _psd(x)
    rms = np.sqrt(np.mean(x ** 2))
    peak = np.max(np.abs(x))
    tot = p.sum() + 1e-12
    pn = p / tot
    out = [
        rms, peak, peak / (rms + 1e-12), stats.kurtosis(x), stats.skew(x),
        p[(f >= 1) & (f < 500)].sum() / tot,      # low band share (unbalance/misalignment)
        p[(f >= 500) & (f < 5000)].sum() / tot,   # high band share (bearing, cavitation)
        (f * pn).sum(),                            # spectral centroid
        -(pn[pn > 0] * np.log2(pn[pn > 0])).sum() / np.log2(len(pn)),  # normalised entropy
        f[np.argmax(p)],
    ]
    if extended:
        def band(fc, hw):
            return p[np.abs(f - fc) <= hw].sum() / tot
        out += [band(fr, 1.5), band(2 * fr, 1.5), band(3 * fr, 1.5),
                band(BPFO_RATIO * fr, 4) + band(2 * BPFO_RATIO * fr, 4),
                band(BPFI_RATIO * fr, 4) + band(2 * BPFI_RATIO * fr, 4)]
    return out


def _unbalance(w):
    rms = np.sqrt(np.mean((w - w.mean(axis=1, keepdims=True)) ** 2, axis=1))
    return 100 * np.max(np.abs(rms - rms.mean())) / (rms.mean() + 1e-12)


def _cur_phase(x):
    f, p = _psd(x)
    band = (f > 15) & (f < 70)  # VFD: fundamental scales with speed
    f0 = f[band][np.argmax(p[band])]
    fund = p[(np.abs(f - f0) <= 1.5)].sum() + 1e-12
    harm = sum(p[(np.abs(f - k * f0) <= 1.5)].sum() for k in range(2, 8))
    side = p[(np.abs(f - f0) > 1.5) & (np.abs(f - f0) <= 15)].sum()
    return np.sqrt(harm / fund), side / fund, f0


def _electrical(cur, volt):
    thd, side, f0 = np.mean([_cur_phase(c) for c in cur], axis=0)
    el = np.array([np.sqrt(np.mean(cur ** 2)), _unbalance(cur), thd, side, _unbalance(volt)])
    return el, f0


def window_features(vib, cur, volt, extended=True):
    """vib (5,n) in g, cur (3,n) in A, volt (3,n) in V -> 21 features (15 if extended=False), order of FEATURE_NAMES."""
    el, f0 = _electrical(cur, volt)
    fr = SLIP_RATIO * f0
    ch = np.mean([_vib_channel(c, fr, extended) for c in vib], axis=0)
    if not extended:
        return np.concatenate([ch, el])
    return np.concatenate([ch[:10], el, [fr], ch[10:]])


def split_windows(sig, size=WINDOW):
    """sig: (channels, n) -> list of (channels, size) windows."""
    return [sig[:, i:i + size] for i in range(0, sig.shape[1] - size + 1, size)]
