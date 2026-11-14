# Is the 28x28 input the bottleneck? Same data, richer image (64 bands x 62 frames, 1 s), 5-class type accuracy.
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import proto
proto.D = "data/esc50_16k"
from proto import (EV, OTHERS, CARS, GENERIC, DISTRACT, SNRS, OUTSIDE, clips, event_windows, bg_windows_cached, mix, NB, codes3, SR, N, H)

NB_BANDS = int(sys.argv[1]) if len(sys.argv) > 1 else 64
binHz = SR / N
ED = np.unique(np.array([min(max(int(150 * np.exp(np.log(7800 / 150) * i / NB_BANDS) / binHz), 1), N // 2) for i in range(NB_BANDS + 1)]))
NBND = len(ED) - 1
WIN = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / (N - 1))).astype(np.float32)
NF = (SR - N) // H + 1
def img_rich(x):
    fr = np.stack([x[f * H:f * H + N] for f in range(NF)]) * WIN
    p = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    out = np.empty((NBND, NF))
    for b in range(NBND):
        out[b] = 10 * np.log10(p[:, ED[b]:max(ED[b + 1], ED[b] + 1)].mean(axis=1) + 1e-10)
    return out

CL = EV + ['other', 'bg']
def codes_rich(img, ref, mode):
    rel = img - ref[:, None]
    nb, nf = rel.shape
    def c3(bits, regions):
        c = np.zeros((nb - 2, nf - 2), dtype=np.int64)
        for db in (-1, 0, 1):
            for dt in (-1, 0, 1):
                c = (c << 1) | bits[1 + db:nb - 1 + db, 1 + dt:nf - 1 + dt]
        reg = (np.arange(1, nb - 1) * regions // nb)[:, None]
        return (reg * 512 + c).ravel()
    if mode == 'relmulti':
        return np.concatenate([c3((rel >= d).astype(np.int64), 14) + k * 14 * 512 for k, d in enumerate((3., 6., 12.))]), 3 * 14 * 512
    up = np.vstack([rel[1:], np.full((1, nf), -1e9)]); dn = np.vstack([np.full((1, nf), -1e9), rel[:-1]])
    pk = (rel > up + 1.0) & (rel > dn + 1.0)
    return np.concatenate([c3((pk & (rel >= 0)).astype(np.int64), 14), c3((pk & (rel >= 6)).astype(np.int64), 14) + 14 * 512,
                           c3((rel >= 6.).astype(np.int64), 14) + 2 * 14 * 512]), 3 * 14 * 512

def softmax_train(X, y, K, epochs=300, lr=0.5, l2=1e-3):
    mu, sd = X.mean(0), X.std(0) + 1e-6; Xn = (X - mu) / sd
    w = np.zeros((Xn.shape[1], K)); b = np.zeros(K)
    cw = np.array([len(y) / (K * max(1, (y == k).sum())) for k in range(K)])
    for _ in range(epochs):
        z = Xn @ w + b; z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
        g = p.copy(); g[np.arange(len(y)), y] -= 1; g *= cw[y][:, None]
        w -= lr * (Xn.T @ g / len(y) + l2 * w); b -= lr * g.mean(0)
    return lambda X2: ((X2 - mu) / sd @ w + b).argmax(1)

res = {}
for ti in (0, 1):
    rng = np.random.default_rng(11 + ti)
    tname, tbg, tlp = CARS[ti]
    train = [c for i, c in enumerate(CARS) if i != ti]
    prof = {c[1]: np.mean([img_rich(mix(w, None, False, 0, 0)).mean(axis=1) for w in bg_windows_cached(c[1], (1, 2))[:80]], axis=0) for c in train}
    data = []   # (label, img, ref)
    def pick(cat, folds):
        l = bg_windows_cached(cat, folds); return l[rng.integers(len(l))]
    for t in EV:
        for f, x in clips(t):
            if f > 2: continue
            for w in event_windows(x):
                for _ in range(2):
                    c = train[rng.integers(3)]
                    data.append((t, img_rich(mix(pick(c[1], (1, 2)), w, t in OUTSIDE, c[2], SNRS[rng.integers(3)])), prof[c[1]]))
    for o in OTHERS:
        for f, x in clips(o):
            if f > 2: continue
            for w in event_windows(x, 2):
                c = train[rng.integers(3)]
                data.append(('other', img_rich(mix(pick(c[1], (1, 2)), w, False, 0, SNRS[rng.integers(3)])), prof[c[1]]))
    for c in train:
        for w in bg_windows_cached(c[1], (1, 2))[::2]: data.append(('bg', img_rich(mix(w, None, False, 0, 0)), prof[c[1]]))
    ref = np.mean([img_rich(mix(w, None, False, 0, 0)).mean(axis=1) for w in bg_windows_cached(tbg, (1, 2, 3))], axis=0)
    tb = bg_windows_cached(tbg, (4, 5))
    tests = []
    for t in EV:
        for f, x in clips(t):
            if f < 4: continue
            for w in event_windows(x):
                for s in (6., 0.):
                    tests.append((t, img_rich(mix(tb[rng.integers(len(tb))], w, t in OUTSIDE, tlp, s))))
    for w in tb: tests.append(('bg', img_rich(mix(w, None, False, 0, 0))))
    for d in DISTRACT:
        for f, x in clips(d):
            if f < 4: continue
            for w in event_windows(x, 2): tests.append(('other', img_rich(mix(tb[rng.integers(len(tb))], w, False, 0, 0))))
    yt = np.array([CL.index(y) for y, _ in tests])
    X = np.array([(img - r[:, None]).ravel() for y, img, r in data]); y = np.array([CL.index(l) for l, _, _ in data])
    clf = softmax_train(X, y, 5)
    pr = clf(np.array([(img - ref[:, None]).ravel() for _, img in tests]))
    res.setdefault('softmax', []).append([np.mean(pr[yt == k] == k) for k in range(5)])
    for mode in ('relmulti', 'peakmulti'):
        F = codes_rich(data[0][1], data[0][2], mode)[1]
        m = {k: NB(F) for k in CL}
        for l, img, r in data: m[l].add(codes_rich(img, r, mode)[0])
        lp = {k: v.logp() for k, v in m.items()}
        pr2 = np.array([CL.index(max(CL, key=lambda k: lp[k][c].mean())) for c in (codes_rich(img, ref, mode)[0] for _, img in tests)])
        res.setdefault('count-' + mode, []).append([np.mean(pr2[yt == k] == k) for k in range(5)])
    print('split', ti, 'done', flush=True)
for k, v in res.items():
    v = np.mean(v, axis=0)
    print(f'{NBND}대역x{NF}프레임 {k:18s} 사이렌 {v[0]*100:3.0f}% 경적 {v[1]*100:3.0f}% 울음 {v[2]*100:3.0f}% 다른소리 {v[3]*100:3.0f}% 배경 {v[4]*100:3.0f}% 평균 {v.mean()*100:3.0f}%')
