# Python replica of the app's counting pipeline, for fast comparison of representations.
# Usage: python -I proto.py data/esc50_16k
import sys, os, time
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "data/esc50_16k"
SR, N, H = 16000, 512, 256
EV = ['siren', 'car_horn', 'crying_baby']
OUTSIDE = {'siren', 'car_horn'}
CARS = [('A', 'engine', 1500.), ('B', 'rain', 2500.), ('C', 'wind', 4000.), ('D', 'train', 2000.)]
GENERIC = ['airplane', 'helicopter']
DISTRACT = ['dog', 'clock_alarm', 'church_bells', 'laughing', 'crow']
OTHERS = ['rooster', 'cat', 'door_wood_knock', 'glass_breaking', 'chainsaw', 'fireworks', 'sneezing', 'coughing',
          'vacuum_cleaner', 'keyboard_typing', 'footsteps', 'clapping', 'chirping_birds', 'thunderstorm', 'hand_saw', 'cow']
SNRS = [6., 0., -6.]
BG_RMS = 0.05

_cache = {}
def clips(cat):
    if cat not in _cache:
        d = os.path.join(D, cat); out = []
        for f in sorted(os.listdir(d)):
            out.append((int(f.split('_')[0]), np.fromfile(os.path.join(d, f), dtype='<i2').astype(np.float32) / 32768))
        _cache[cat] = out
    return _cache[cat]

def rms(x): return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))
def windows(x): return [x[i:i + SR] for i in range(0, len(x) - SR + 1, SR // 2)]
def event_windows(x, mx=4):
    w = windows(x); r = np.array([rms(v) for v in w]); top = r.max()
    idx = [i for i in np.argsort(-r) if r[i] >= top * 10 ** (-10 / 20)][:mx]
    return [w[i] for i in idx]
def bg_windows(cat, folds): return [w for f, x in clips(cat) if f in folds for w in windows(x) if rms(w) > 1e-4]

def lowpass(x, fc):
    a = np.exp(-2 * np.pi * fc / SR)
    from scipy.signal import lfilter
    y = lfilter([1 - a], [1, -a], x); return lfilter([1 - a], [1, -a], y).astype(np.float32)

def mix(bg, ev, outside, lp, snr, gain_db=0.):
    out = bg * (BG_RMS / rms(bg))
    if ev is not None:
        e = lowpass(ev, lp) if outside else ev
        out = out + e * (BG_RMS * 10 ** (snr / 20) / rms(e))
    out = out * 10 ** (gain_db / 20)
    return np.clip(np.round(np.clip(out, -1, 1) * 32000), -32768, 32767) / 32768.

# ---------- app's band image ----------
binHz = SR / N
EDGES = np.array([min(max(int(150 * np.exp(np.log(7800 / 150) * i / 28) / binHz), 1), N // 2) for i in range(29)])
WIN = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / (N - 1))).astype(np.float32)
NF = (SR - N) // H + 1
def band_image(x):
    fr = np.stack([x[f * H:f * H + N] for f in range(NF)]) * WIN
    p = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    bands = np.empty((28, NF))
    for b in range(28):
        lo, hi = EDGES[b], max(EDGES[b + 1], EDGES[b] + 1)
        bands[b] = 10 * np.log10(p[:, lo:hi].mean(axis=1) + 1e-10)
    cols = np.empty((28, 28))
    for c in range(28):
        a = c * NF // 28; z = max(a + 1, (c + 1) * NF // 28)
        cols[:, c] = bands[:, a:min(z, NF)].mean(axis=1)
    return cols  # [band][time]

# ---------- encodings -> code index arrays ----------
SH = [(db, dt) for db in (-1, 0, 1) for dt in (-1, 0, 1)]
def codes3(bits, regions):
    c = np.zeros((26, 26), dtype=np.int64)
    for db, dt in SH:
        c = (c << 1) | bits[1 + db:27 + db, 1 + dt:27 + dt]
    reg = (np.arange(1, 27) * regions // 28)[:, None]
    return (reg * 512 + c).ravel()

def encode(img, mode, ref=None):
    rel = img - ref[:, None] if ref is not None else img
    q = lambda p: np.quantile(rel, p)
    if mode == 'base':
        return codes3((rel >= q(0.70)).astype(np.int64), 7), 7 * 512
    if mode == 'reg28':
        return codes3((rel >= q(0.70)).astype(np.int64), 28), 28 * 512
    if mode == 'multi':
        out = [codes3((rel >= q(p)).astype(np.int64), 7) + k * 7 * 512 for k, p in enumerate((0.9, 0.7, 0.5))]
        return np.concatenate(out), 3 * 7 * 512
    if mode == 'multi14':
        out = [codes3((rel >= q(p)).astype(np.int64), 14) + k * 14 * 512 for k, p in enumerate((0.9, 0.7, 0.5))]
        return np.concatenate(out), 3 * 14 * 512
    if mode in ('peak', 'peakrel', 'peakmulti'):
        # spectral peak map: cell higher than the bands just above and below
        up = np.vstack([rel[1:], np.full((1, rel.shape[1]), -1e9)]); dn = np.vstack([np.full((1, rel.shape[1]), -1e9), rel[:-1]])
        pk = (rel > up + 1.0) & (rel > dn + 1.0)
        if mode == 'peak':
            return codes3(pk.astype(np.int64), 7), 7 * 512
        if mode == 'peakrel':   # peak AND above this car's background by 3 dB
            return codes3((pk & (rel >= 3.)).astype(np.int64), 7), 7 * 512
        out = [codes3((pk & (rel >= d)).astype(np.int64), 7) + k * 7 * 512 for k, d in enumerate((0., 6.))]
        out.append(codes3((rel >= 6.).astype(np.int64), 7) + 2 * 7 * 512)
        return np.concatenate(out), 3 * 7 * 512
    if mode == 'rel3':
        return codes3((rel >= max(q(0.70), 3.)).astype(np.int64), 7), 7 * 512
    if mode == 'relabs6':   # only "above this car's background by 6 dB" (no top-30% cut)
        return codes3((rel >= 6.).astype(np.int64), 7), 7 * 512
    if mode == 'relmulti':  # 3, 6, 12 dB above background
        out = [codes3((rel >= d).astype(np.int64), 7) + k * 7 * 512 for k, d in enumerate((3., 6., 12.))]
        return np.concatenate(out), 3 * 7 * 512
    if mode == 'rel6':   # needs ref
        return codes3((rel >= max(q(0.70), 6.)).astype(np.int64), 7), 7 * 512
    raise ValueError(mode)

class NB:
    def __init__(self, F): self.c = np.zeros(F); self.F = F
    def add(self, codes): np.add.at(self.c, codes, 1)
    def logp(self):
        regs = self.F // 512
        tot = self.c.reshape(regs, 512).sum(axis=1)
        return np.log((self.c + 0.5) / (np.repeat(tot, 512) + 0.5 * 512))
def score(lp, codes): return lp[codes].mean()

# ---------- split ----------
def split(ti, seed):
    rng = np.random.default_rng(seed)
    tname, tbg, tlp = CARS[ti]
    train = [c for i, c in enumerate(CARS) if i != ti]
    def pick(cat, folds):
        l = bg_windows_cached(cat, folds); return l[rng.integers(len(l))]
    S = {}
    S['trainEv'] = {t: [] for t in EV}
    for t in EV:
        for f, x in clips(t):
            if f > 2: continue
            for w in event_windows(x):
                for _ in range(2):
                    c = train[rng.integers(3)]
                    S['trainEv'][t].append((band_image(mix(pick(c[1], (1, 2)), w, t in OUTSIDE, c[2], SNRS[rng.integers(3)])), c[1]))
    S['trainBg'] = [(band_image(mix(w, None, False, 0, 0)), c[1]) for c in train for w in bg_windows_cached(c[1], (1, 2))] + \
                   [(band_image(mix(w, None, False, 0, 0)), g) for g in GENERIC for w in bg_windows_cached(g, (1, 2))]
    S['trainOther'] = []
    for o in OTHERS:
        for f, x in clips(o):
            if f > 2: continue
            for w in event_windows(x, 2):
                c = train[rng.integers(3)]
                S['trainOther'].append((band_image(mix(pick(c[1], (1, 2)), w, False, 0, SNRS[rng.integers(3)])), c[1]))
    S['calib'] = [band_image(mix(w, None, False, 0, 0)) for c in train for w in bg_windows_cached(c[1], (3,))]
    for o in OTHERS:
        for f, x in clips(o):
            if f != 3: continue
            for w in event_windows(x, 1):
                c = train[rng.integers(3)]
                S['calib'].append(band_image(mix(pick(c[1], (3,)), w, False, 0, 0)))
    S['carStream'] = [band_image(mix(w, None, False, 0, 0)) for w in bg_windows_cached(tbg, (1, 2, 3))]
    S['confirmed'] = {t: [] for t in EV}
    for t in EV:
        for f, x in [c for c in clips(t) if c[0] == 3][:5]:
            for w in event_windows(x, 3):
                S['confirmed'][t].append(band_image(mix(pick(tbg, (3,)), w, t in OUTSIDE, tlp, SNRS[rng.integers(3)])))
    tb = bg_windows_cached(tbg, (4, 5))
    S['testEv'] = {t: [] for t in EV}
    for t in EV:
        for f, x in clips(t):
            if f < 4: continue
            for w in event_windows(x):
                for s in SNRS:
                    S['testEv'][t].append((s, band_image(mix(tb[rng.integers(len(tb))], w, t in OUTSIDE, tlp, s))))
    S['testBg'] = [band_image(mix(w, None, False, 0, 0)) for w in tb]
    S['testDis'] = []
    for d in DISTRACT:
        for f, x in clips(d):
            if f < 4: continue
            for w in event_windows(x, 2):
                S['testDis'].append(band_image(mix(tb[rng.integers(len(tb))], w, False, 0, 0)))
    # event-level: whole 5 s clips; event scaled by its loudest window
    S['clipEv'] = {t: [] for t in EV}
    bgclips = [x for f, x in clips(tbg) if f >= 4]
    for t in EV:
        for f, x in clips(t):
            if f < 4: continue
            top = max(rms(w) for w in windows(x))
            for s in SNRS:
                b = bgclips[rng.integers(len(bgclips))]
                e = lowpass(x, tlp) if t in OUTSIDE else x
                etop = max(rms(w) for w in windows(e))
                m = b * (BG_RMS / rms(b)) + e * (BG_RMS * 10 ** (s / 20) / etop)
                m = np.clip(np.round(np.clip(m, -1, 1) * 32000), -32768, 32767) / 32768.
                S['clipEv'][t].append((s, [band_image(w) for w in windows(m)]))
    S['clipBg'] = [[band_image(w * (BG_RMS / rms(x))) for w in windows(x)] for x in bgclips]
    S['clipDis'] = []
    for d in DISTRACT:
        for f, x in clips(d):
            if f < 4: continue
            b = bgclips[rng.integers(len(bgclips))]
            m = b * (BG_RMS / rms(b)) + x * (BG_RMS / max(rms(w) for w in windows(x)))
            S['clipDis'].append([band_image(w) for w in windows(np.clip(m, -1, 1))])
    S['target'] = tbg
    return S

_bgc = {}
def bg_windows_cached(cat, folds):
    k = (cat, tuple(sorted(folds)))
    if k not in _bgc: _bgc[k] = bg_windows(cat, set(folds))
    return _bgc[k]

if __name__ == '__main__':
    import pickle
    out = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    for i in range(4):
        t = time.time(); S = split(i, 100 + i)
        pickle.dump(S, open(os.path.join(out, f'split{i}.pkl'), 'wb'))
        print('split', i, CARS[i][1], 'done', round(time.time() - t, 1), 's', {t: len(v) for t, v in S['testEv'].items()}, len(S['testBg']), len(S['testDis']))
