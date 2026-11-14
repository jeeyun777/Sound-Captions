# Siren frequency-track features from one 1 s window (16 kHz): dominant pitch in 400-1800 Hz per frame
import numpy as np
SR, N, H = 16000, 1024, 160
_W = np.hanning(N).astype(np.float32)
_f = np.fft.rfftfreq(N, 1 / SR)
_band = np.where((_f >= 400) & (_f <= 1800))[0]

def fm_features(x):
    n = 1 + (len(x) - N) // H
    fr = np.stack([x[i * H:i * H + N] for i in range(n)]) * _W
    p = np.abs(np.fft.rfft(fr, axis=1)) ** 2 + 1e-12
    pb = p[:, _band]
    k = pb.argmax(1)
    # parabolic interpolation of the peak for finer frequency
    lo = np.clip(k - 1, 0, len(_band) - 1); hi = np.clip(k + 1, 0, len(_band) - 1)
    a, b, c = np.log(pb[np.arange(n), lo]), np.log(pb[np.arange(n), k]), np.log(pb[np.arange(n), hi])
    d = np.where(np.abs(a - 2 * b + c) > 1e-9, 0.5 * (a - c) / (a - 2 * b + c), 0.0)
    freq = _f[_band][k] + d * (SR / N)
    prom = 10 * np.log10(pb.max(1) / np.median(pb, axis=1))            # peak vs rest of the band (dB)
    harm = np.array([p[i, min(int(round(2 * freq[i] / (SR / N))), p.shape[1] - 1)] / pb[i].max() for i in range(n)])
    tonal = prom > 12
    semi = 12 * np.log2(np.maximum(freq, 1) / 440.0)
    ds = np.abs(np.diff(semi))
    both = tonal[1:] & tonal[:-1]
    cont = (ds[both] < 0.6).mean() if both.any() else 0.0          # glides smoothly between frames
    st = semi[tonal]
    rng_ = (np.percentile(st, 95) - np.percentile(st, 5)) if tonal.sum() > 3 else 0.0   # how far it sweeps
    slope = np.diff(st) if tonal.sum() > 3 else np.zeros(1)
    move = np.mean(np.abs(slope)) if len(slope) else 0.0
    # direction changes per second (yelp/hi-lo are fast, wail slow)
    sg = np.sign(np.convolve(np.diff(semi), np.ones(5) / 5, 'same'))
    turns = np.sum(np.abs(np.diff(sg[tonal[1:]])) > 0) if tonal.sum() > 6 else 0
    return np.array([tonal.mean(), np.median(prom), cont, rng_, move, float(turns), float(np.log10(np.median(harm) + 1e-6))])
