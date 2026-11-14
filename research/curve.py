# Learning curve: does type accuracy keep rising with more training clips? (counting relmulti, car-relative)
import sys, os, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proto import (EV, OTHERS, CARS, GENERIC, SNRS, OUTSIDE, clips, event_windows, bg_windows_cached, mix,
                   band_image, encode, NB)

SP = sys.argv[1]
import proto; proto.D = "data/esc50_16k"
CL = EV + ['other', 'bg']
MIXES = int(sys.argv[2]) if len(sys.argv) > 2 else 4
res = {}
for ti in (0, 1):
    S = pickle.load(open(os.path.join(SP, f'split{ti}.pkl'), 'rb'))
    rng = np.random.default_rng(7 + ti)
    train = [c for i, c in enumerate(CARS) if i != ti]
    prof = {c[1]: np.mean([band_image(mix(w, None, False, 0, 0)).mean(axis=1) for w in bg_windows_cached(c[1], (1, 2))[:60]], axis=0) for c in train}
    ref = np.mean([img.mean(axis=1) for img in S['carStream']], axis=0)
    tests = [(EV[k], img) for k, t in enumerate(EV) for s, img in S['testEv'][t] if s >= 0] + \
            [('bg', img) for img in S['testBg']] + [('other', img) for img in S['testDis']]
    tcodes = [(y, encode(img, 'relmulti', ref)[0]) for y, img in tests]
    F = 3 * 7 * 512
    bgm = NB(F)
    for c in train:
        for w in bg_windows_cached(c[1], (1, 2)): bgm.add(encode(band_image(mix(w, None, False, 0, 0)), 'relmulti', prof[c[1]])[0])
    for folds in ((1,), (1, 2), (1, 2, 3)):
        m = {t: NB(F) for t in EV}; m['other'] = NB(F); m['bg'] = bgm
        def add(model, x, outside, lp_src):
            c = train[rng.integers(3)]
            bgw = bg_windows_cached(c[1], (1, 2)); b = bgw[rng.integers(len(bgw))]
            img = band_image(mix(b, x, outside, c[2], SNRS[rng.integers(3)]))
            model.add(encode(img, 'relmulti', prof[c[1]])[0])
        for t in EV:
            for f, x in clips(t):
                if f not in folds: continue
                for w in event_windows(x):
                    for _ in range(MIXES): add(m[t], w, t in OUTSIDE, None)
        for o in OTHERS:
            for f, x in clips(o):
                if f not in folds: continue
                for w in event_windows(x, 2): add(m['other'], w, False, None)
        lp = {k: v.logp() for k, v in m.items()}
        acc = {k: [] for k in CL}
        for y, c in tcodes: acc[y].append(max(CL, key=lambda k: lp[k][c].mean()) == y)
        res.setdefault(len(folds) * 8, []).append([np.mean(acc[k]) for k in CL])
for n, v in sorted(res.items()):
    v = np.mean(v, axis=0)
    print(f'종류당 학습 {n:2d}개 클립 (x{MIXES} 혼합): 사이렌 {v[0]*100:3.0f}% 경적 {v[1]*100:3.0f}% 울음 {v[2]*100:3.0f}% 다른소리 {v[3]*100:3.0f}% 배경 {v[4]*100:3.0f}% 평균 {v.mean()*100:3.0f}%', flush=True)
