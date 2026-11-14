# Two stages: (1) band baseline detects "something above this car's usual sound",
# (2) counting decides what it is (siren / horn / baby / other / background).
import sys, os, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proto import encode, NB, EV, SNRS

def auc(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    allv = np.concatenate([pos, neg]); r = allv.argsort().argsort() + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

SP = sys.argv[1]
MODES = sys.argv[2].split(',')
splits = [pickle.load(open(os.path.join(SP, f'split{i}.pkl'), 'rb')) for i in range(4)]

def excess(img, ref, k=5):
    """how far the loudest moments of the most raised band stand above this car's background (dB)"""
    return float(np.max(np.sort(img - ref[:, None], axis=1)[:, -k:].mean(axis=1)))

class Hybrid:
    def __init__(self, S, mode, adapt=True, conf=False):
        self.mode = mode
        rel = mode.startswith('rel')
        prof = {}
        for img, tag in S['trainBg']: prof.setdefault(tag, []).append(img.mean(axis=1))
        prof = {k: np.mean(v, axis=0) for k, v in prof.items()}
        factory_ref = np.mean([img.mean(axis=1) for img, _ in S['trainBg']], axis=0)
        enc = lambda img, ref: encode(img, mode, ref if rel else None)[0]
        F = encode(S['trainBg'][0][0], mode, factory_ref if rel else None)[1]
        self.nb = {}
        for t in EV:
            m = NB(F); [m.add(enc(i, prof[g])) for i, g in S['trainEv'][t]]; self.nb[t] = m
        self.nb['bg'] = NB(F); [self.nb['bg'].add(enc(i, prof[g])) for i, g in S['trainBg']]
        self.nb['other'] = NB(F); [self.nb['other'].add(enc(i, prof[g])) for i, g in S['trainOther']]
        if adapt:   # this car: own background profile, threshold from its own usual sound (no labels)
            self.ref = np.mean([img.mean(axis=1) for img in S['carStream']], axis=0)
            ex = [excess(i, self.ref) for i in S['carStream']]
            self.det = float(np.quantile(ex, 0.99))
            for i in S['carStream']: self.nb['bg'].add(enc(i, self.ref))
        else:       # factory: average training background, threshold from training cars
            self.ref = factory_ref
            self.det = float(np.quantile([excess(i, factory_ref) for i in S['calib'][:len(S['calib']) // 2]], 0.99))
        if conf:    # events confirmed by the driver's reaction in this car
            for t in EV:
                for i in S['confirmed'][t]: self.nb[t].add(enc(i, self.ref))
        self.lp = {k: m.logp() for k, m in self.nb.items()}
        self.enc = lambda img: enc(img, self.ref)
    def decide(self, img):
        e = excess(img, self.ref)
        c = self.enc(img)
        s = {k: lp[c].mean() for k, lp in self.lp.items()}
        best = max(s, key=s.get)
        return e, best, s
    def margin(self, img):
        """single score for AUC: detection excess if pattern says event, else very low"""
        e, best, s = self.decide(img)
        ev = max(EV, key=lambda t: s[t])
        return ev, (e - self.det) + 2.0 * (s[ev] - max(s['other'], s['bg'])) / 0.05
    def hit(self, img, t):
        e, best, s = self.decide(img)
        if e <= self.det or best not in EV: return False
        return best == t if t else True

def evaluate(h, S):
    rec = {t: np.mean([h.hit(i, t) for s, i in S['testEv'][t]]) for t in EV}
    bysnr = [np.mean([h.hit(i, t) for t in EV for s, i in S['testEv'][t] if s == q]) for q in SNRS]
    fab = np.mean([h.hit(i, None) for i in S['testBg']]); fad = np.mean([h.hit(i, None) for i in S['testDis']])
    # type accuracy among windows that pass detection
    det_ok = [(t, h.decide(i)) for t in EV for s, i in S['testEv'][t]]
    passed = [(t, b) for t, (e, b, s) in det_ok if e > h.det]
    typeacc = np.mean([b == t for t, b in passed]) if passed else 0
    def ev_det(wins, t):
        hh = [h.hit(w, t) for w in wins]
        return any(sum(hh[k:k + 3]) >= 2 for k in range(len(hh) - 1))
    erec = {t: np.mean([ev_det(w, t) for s, w in S['clipEv'][t]]) for t in EV}
    erec_snr = [np.mean([ev_det(w, t) for t in EV for s, w in S['clipEv'][t] if s == q]) for q in SNRS]
    efab = np.mean([ev_det(w, None) for w in S['clipBg']]); efad = np.mean([ev_det(w, None) for w in S['clipDis']])
    detpass = np.mean([e > h.det for t, (e, b, s) in det_ok])
    return dict(rec=rec, bysnr=bysnr, fab=fab, fad=fad, typeacc=typeacc, detpass=detpass, erec=erec, erec_snr=erec_snr, efab=efab, efad=efad)

def avg(rs):
    out = {}
    for key, v in rs[0].items():
        if isinstance(v, dict): out[key] = {t: np.mean([r[key][t] for r in rs]) for t in v}
        elif isinstance(v, list): out[key] = list(np.mean([r[key] for r in rs], axis=0))
        else: out[key] = np.mean([r[key] for r in rs])
    return out

def show(name, a):
    print(f"{name:24s} 감지통과 {a['detpass']*100:3.0f}% 종류정확 {a['typeacc']*100:3.0f}% | 창 사{a['rec']['siren']*100:3.0f} 경{a['rec']['car_horn']*100:3.0f} 울{a['rec']['crying_baby']*100:3.0f} "
          f"(SNR {'/'.join(f'{x*100:.0f}' for x in a['bysnr'])}) 오경보 {a['fab']*100:4.1f}/{a['fad']*100:4.1f}% | "
          f"사건 사{a['erec']['siren']*100:3.0f} 경{a['erec']['car_horn']*100:3.0f} 울{a['erec']['crying_baby']*100:3.0f} "
          f"(SNR {'/'.join(f'{x*100:.0f}' for x in a['erec_snr'])}) 사건오경보 {a['efab']*100:4.1f}/{a['efad']*100:4.1f}%", flush=True)

for mode in MODES:
    for adapt, conf, tag in ((False, False, 'S1 출고'), (True, False, 'S2 차 적응'), (True, True, 'S3 +반응학습')):
        show(f'{mode} {tag}', avg([evaluate(Hybrid(S, mode, adapt, conf), S) for S in splits]))
