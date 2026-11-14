# How much "what is it" information is in the 28x28 band image? (5 classes, test car never seen)
import sys, os, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proto import EV, encode, NB

SP = sys.argv[1]
CL = EV + ['other', 'bg']

def feats(img, ref): return (img - ref[:, None]).ravel()

def softmax_train(X, y, K, epochs=300, lr=0.5, l2=1e-3, hidden=0, seed=0):
    rng = np.random.default_rng(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6; Xn = (X - mu) / sd
    w = np.zeros((Xn.shape[1], K)); b = np.zeros(K)
    cw = np.array([len(y) / (K * max(1, (y == k).sum())) for k in range(K)])
    for _ in range(epochs):
        z = Xn @ w + b; z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
        g = p.copy(); g[np.arange(len(y)), y] -= 1; g *= cw[y][:, None]
        w -= lr * (Xn.T @ g / len(y) + l2 * w); b -= lr * g.mean(0)
    return lambda X2: ((X2 - mu) / sd @ w + b).argmax(1)

def nb_pred(S, ref, prof, mode, skip_flat):
    F = encode(S['trainBg'][0][0], mode, ref)[1]
    m = {c: NB(F) for c in CL}
    for t in EV:
        for i, g in S['trainEv'][t]: m[t].add(encode(i, mode, prof[g])[0])
    for i, g in S['trainOther']: m['other'].add(encode(i, mode, prof[g])[0])
    for i, g in S['trainBg']: m['bg'].add(encode(i, mode, prof[g])[0])
    lp = {c: x.logp() for c, x in m.items()}
    def pred(img):
        c = encode(img, mode, ref)[0]
        if skip_flat:
            pat = c % 512; keep = (pat != 0) & (pat != 511)
            if keep.sum() > 10: c = c[keep]
        return max(CL, key=lambda k: lp[k][c].mean())
    return pred

res = {}
for i in range(4):
    S = pickle.load(open(os.path.join(SP, f'split{i}.pkl'), 'rb'))
    prof = {}
    for img, tag in S['trainBg']: prof.setdefault(tag, []).append(img.mean(axis=1))
    prof = {k: np.mean(v, axis=0) for k, v in prof.items()}
    ref = np.mean([img.mean(axis=1) for img in S['carStream']], axis=0)
    X, y = [], []
    for k, t in enumerate(EV):
        for img, g in S['trainEv'][t]: X.append(feats(img, prof[g])); y.append(k)
    for img, g in S['trainOther']: X.append(feats(img, prof[g])); y.append(3)
    for img, g in S['trainBg']: X.append(feats(img, prof[g])); y.append(4)
    X = np.array(X); y = np.array(y)
    clf = softmax_train(X, y, 5) if 'softmax' not in res or len(res['softmax']) < 4 else None
    tests = [(k, img) for k, t in enumerate(EV) for s, img in S['testEv'][t] if s >= 0] + \
            [(4, img) for img in S['testBg']] + [(3, img) for img in S['testDis']]
    pr = clf(np.array([feats(img, ref) for k, img in tests]))
    yt = np.array([k for k, _ in tests])
    res.setdefault('softmax', []).append([np.mean(pr[yt == k] == k) for k in range(5)])
    for mode, skip in (('relmulti', False), ('peak', False), ('peakrel', False), ('peakmulti', False)):
        p = nb_pred(S, ref if mode != 'peak' else None, prof, mode, skip)
        pr2 = np.array([CL.index(p(img)) for k, img in tests])
        res.setdefault(f'count-{mode}{"-active" if skip else ""}', []).append([np.mean(pr2[yt == k] == k) for k in range(5)])
for k, v in res.items():
    v = np.mean(v, axis=0)
    print(f'{k:24s} 사이렌 {v[0]*100:3.0f}% 경적 {v[1]*100:3.0f}% 울음 {v[2]*100:3.0f}% 다른소리 {v[3]*100:3.0f}% 배경 {v[4]*100:3.0f}%  평균 {v.mean()*100:3.0f}%')
