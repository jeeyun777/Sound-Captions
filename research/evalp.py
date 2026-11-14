import sys, os, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proto import encode, NB, EV, SNRS

SP = sys.argv[1]
MODES = sys.argv[2].split(',')
splits = [pickle.load(open(os.path.join(SP, f'split{i}.pkl'), 'rb')) for i in range(4)]

def auc(pos, neg):
    pos = np.asarray(pos); neg = np.asarray(neg)
    allv = np.concatenate([pos, neg]); r = allv.argsort().argsort() + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

class CountClf:
    """the app's decision: best event model minus max(background, other)"""
    def __init__(self, S, mode, relative, other=True, carRef=False):
        self.mode, self.rel = mode, relative
        prof = {}
        if relative:
            for img, tag in S['trainBg']: prof.setdefault(tag, []).append(img.mean(axis=1))
            prof = {k: np.mean(v, axis=0) for k, v in prof.items()}
            self.ref = np.mean([img.mean(axis=1) for img, _ in S['trainBg']], axis=0)
            if carRef: self.ref = np.mean([img.mean(axis=1) for img in S['carStream']], axis=0)   # this car's own background
        enc = lambda img, tag: encode(img, mode, prof.get(tag) if relative else None)
        F = enc(S['trainBg'][0][0], S['trainBg'][0][1])[1]
        self.m = {}
        for t in EV:
            nb = NB(F); [nb.add(enc(i, g)[0]) for i, g in S['trainEv'][t]]; self.m[t] = nb
        self.bg = NB(F); [self.bg.add(enc(i, g)[0]) for i, g in S['trainBg']]
        self.oth = None
        if other:
            self.oth = NB(F); [self.oth.add(enc(i, g)[0]) for i, g in S['trainOther']]
        self.refresh()
    def refresh(self):
        self.lp = {t: m.logp() for t, m in self.m.items()}; self.lbg = self.bg.logp()
        self.loth = self.oth.logp() if self.oth is not None else None
    def codes(self, img): return encode(img, self.mode, self.ref if self.rel else None)[0]
    def margin(self, img):
        c = self.codes(img)
        sb = self.lbg[c].mean()
        if self.loth is not None: sb = max(sb, self.loth[c].mean())
        s = {t: self.lp[t][c].mean() - sb for t in EV}
        b = max(s, key=s.get); return b, s[b]

def evaluate(clf, S, thr=None):
    if thr is None:
        ms = [clf.margin(i) for i in S['calib']]
        thr = {t: (np.quantile([m for b, m in ms if b == t], 0.99) if any(b == t for b, _ in ms) else 0.0) for t in EV}
    hit = lambda img, t: (lambda b, m: (b == t if t else True) and m > thr[b])(*clf.margin(img))
    rec = {t: np.mean([hit(i, t) for s, i in S['testEv'][t]]) for t in EV}
    bysnr = [np.mean([hit(i, t) for t in EV for s, i in S['testEv'][t] if s == q]) for q in SNRS]
    pos = [clf.margin(i)[1] for t in EV for s, i in S['testEv'][t]]
    nb_ = [clf.margin(i)[1] for i in S['testBg']]; nd = [clf.margin(i)[1] for i in S['testDis']]
    fab = np.mean([hit(i, None) for i in S['testBg']]); fad = np.mean([hit(i, None) for i in S['testDis']])
    # event level: 2 of any 3 consecutive windows, correct type
    def ev_det(wins, t):
        h = [hit(w, t) for w in wins]
        return any(sum(h[k:k + 3]) >= 2 for k in range(len(h) - 1))
    erec = {t: np.mean([ev_det(w, t) for s, w in S['clipEv'][t]]) for t in EV}
    erec_snr = [np.mean([ev_det(w, t) for t in EV for s, w in S['clipEv'][t] if s == q]) for q in SNRS]
    efa_bg = np.mean([ev_det(w, None) for w in S['clipBg']]); efa_dis = np.mean([ev_det(w, None) for w in S['clipDis']])
    return dict(rec=rec, bysnr=bysnr, aucb=auc(pos, nb_), aucd=auc(pos, nd), fab=fab, fad=fad, erec=erec, erec_snr=erec_snr, efab=efa_bg, efad=efa_dis)

def avg(rs):
    k = rs[0]
    out = {}
    for key, v in k.items():
        if isinstance(v, dict): out[key] = {t: np.mean([r[key][t] for r in rs]) for t in v}
        elif isinstance(v, list): out[key] = list(np.mean([r[key] for r in rs], axis=0))
        else: out[key] = np.mean([r[key] for r in rs])
    return out

def show(name, a):
    print(f"{name:28s} 창 재현율 사{a['rec']['siren']*100:3.0f} 경{a['rec']['car_horn']*100:3.0f} 울{a['rec']['crying_baby']*100:3.0f} "
          f"(SNR {'/'.join(f'{x*100:.0f}' for x in a['bysnr'])}) 오경보 {a['fab']*100:4.1f}/{a['fad']*100:4.1f}% AUC {a['aucb']:.3f}/{a['aucd']:.3f} | "
          f"사건 재현율 사{a['erec']['siren']*100:3.0f} 경{a['erec']['car_horn']*100:3.0f} 울{a['erec']['crying_baby']*100:3.0f} "
          f"(SNR {'/'.join(f'{x*100:.0f}' for x in a['erec_snr'])}) 사건 오경보 {a['efab']*100:4.1f}/{a['efad']*100:4.1f}%", flush=True)

for mode in MODES:
    car = mode.endswith('car'); base = mode[:-3] if car else mode
    rel = base.startswith('rel')
    rs = [evaluate(CountClf(S, base, rel, carRef=car), S) for S in splits]
    show(mode, avg(rs))
