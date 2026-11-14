# Build one leave-one-car-out split with waveforms -> (app band image, YAMNet scores, YAMNet embedding) per window.
# Usage: python -I ybuild.py <split index> <out dir>
import sys, os, time, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import proto
proto.D = "data/esc50_16k"
from proto import (EV, OUTSIDE, CARS, GENERIC, DISTRACT, OTHERS, SNRS, BG_RMS, clips, rms, windows, event_windows,
                   bg_windows_cached, lowpass, mix, band_image)
from yamnet_np import YAMNet

ti, out = int(sys.argv[1]), sys.argv[2]
rng = np.random.default_rng(100 + ti)
tname, tbg, tlp = CARS[ti]
train = [c for i, c in enumerate(CARS) if i != ti]
def pick(cat, folds):
    l = bg_windows_cached(cat, folds); return l[rng.integers(len(l))]

W = []          # all waveforms; the split stores indices into W
def add(w): W.append(w.astype(np.float32)); return len(W) - 1

S = {'target': tbg}
S['trainEv'] = {t: [] for t in EV}
for t in EV:
    for f, x in clips(t):
        if f > 2: continue
        for w in event_windows(x):
            for _ in range(2):
                c = train[rng.integers(3)]
                S['trainEv'][t].append((add(mix(pick(c[1], (1, 2)), w, t in OUTSIDE, c[2], SNRS[rng.integers(3)])), c[1]))
S['trainBg'] = [(add(mix(w, None, False, 0, 0)), c[1]) for c in train for w in bg_windows_cached(c[1], (1, 2))[::3]] + \
               [(add(mix(w, None, False, 0, 0)), g) for g in GENERIC for w in bg_windows_cached(g, (1, 2))[::3]]
S['trainOther'] = []
for o in OTHERS:
    for f, x in clips(o):
        if f > 2: continue
        for w in event_windows(x, 2):
            c = train[rng.integers(3)]
            S['trainOther'].append((add(mix(pick(c[1], (1, 2)), w, False, 0, SNRS[rng.integers(3)])), c[1]))
S['calib'] = [add(mix(w, None, False, 0, 0)) for c in train for w in bg_windows_cached(c[1], (3,))]
for o in OTHERS:
    for f, x in clips(o):
        if f != 3: continue
        for w in event_windows(x, 1):
            c = train[rng.integers(3)]
            S['calib'].append(add(mix(pick(c[1], (3,)), w, False, 0, 0)))
S['carStream'] = [add(mix(w, None, False, 0, 0)) for w in bg_windows_cached(tbg, (1, 2, 3))]
S['confirmed'] = {t: [] for t in EV}
for t in EV:
    for f, x in [c for c in clips(t) if c[0] == 3][:5]:
        for w in event_windows(x, 3):
            S['confirmed'][t].append(add(mix(pick(tbg, (3,)), w, t in OUTSIDE, tlp, SNRS[rng.integers(3)])))
S['wrong'] = [add(mix(pick(tbg, (3,)), w, True, tlp, 0.)) for f, x in clips('car_horn') if f == 3 for w in event_windows(x, 4)]
tb = bg_windows_cached(tbg, (4, 5))
S['testEv'] = {t: [] for t in EV}
for t in EV:
    for f, x in clips(t):
        if f < 4: continue
        for w in event_windows(x):
            for s in SNRS:
                S['testEv'][t].append((s, add(mix(tb[rng.integers(len(tb))], w, t in OUTSIDE, tlp, s))))
S['testBg'] = [add(mix(w, None, False, 0, 0)) for w in tb]
S['testDis'] = [add(mix(tb[rng.integers(len(tb))], w, False, 0, 0)) for d in DISTRACT for f, x in clips(d) if f >= 4 for w in event_windows(x, 2)]
# mic gain variants of the test set (+-12 dB)
S['testGain'] = {}
for g in (12., -12.):
    S['testGain'][g] = {'ev': [(t, s, add(np.clip(W[i] * 10 ** (g / 20), -1, 1))) for t in EV for s, i in S['testEv'][t]],
                        'bg': [add(np.clip(W[i] * 10 ** (g / 20), -1, 1)) for i in S['testBg']],
                        'dis': [add(np.clip(W[i] * 10 ** (g / 20), -1, 1)) for i in S['testDis']]}
# event level: whole 5 s clips
bgclips = [x for f, x in clips(tbg) if f >= 4]
S['clipEv'] = {t: [] for t in EV}
for t in EV:
    for f, x in clips(t):
        if f < 4: continue
        for s in SNRS:
            b = bgclips[rng.integers(len(bgclips))]
            e = lowpass(x, tlp) if t in OUTSIDE else x
            etop = max(rms(w) for w in windows(e))
            m = b * (BG_RMS / rms(b)) + e * (BG_RMS * 10 ** (s / 20) / etop)
            m = np.clip(np.round(np.clip(m, -1, 1) * 32000), -32768, 32767) / 32768.
            S['clipEv'][t].append((s, [add(w) for w in windows(m)]))
S['clipBg'] = [[add(w * (BG_RMS / rms(x))) for w in windows(x)] for x in bgclips]
S['clipDis'] = []
for d in DISTRACT:
    for f, x in [c for c in clips(d) if c[0] >= 4][:8]:
        b = bgclips[rng.integers(len(bgclips))]
        m = b * (BG_RMS / rms(b)) + x * (BG_RMS / max(rms(w) for w in windows(x)))
        S['clipDis'].append([add(w) for w in windows(np.clip(m, -1, 1).astype(np.float32))])

t0 = time.time()
S['img'] = np.stack([band_image(w) for w in W]).astype(np.float32)
m = YAMNet('data/yamnet/yamnet.h5')
sc, em = m.run_windows(W, batch=128)
S['ysc'] = sc.astype(np.float32); S['yem'] = em.astype(np.float32)
pickle.dump(S, open(os.path.join(out, f'ysplit{ti}.pkl'), 'wb'))
print('split', ti, tbg, 'windows', len(W), 'features %.0f s' % (time.time() - t0), flush=True)
