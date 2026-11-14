# Step-by-step: YAMNet alone -> + car background detection -> counting on YAMNet outputs -> car adaptation -> reaction learning -> self test
import sys, os, pickle, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
EV = ['siren', 'car_horn', 'crying_baby']
SNRS = [6., 0., -6.]
CL = EV + ['other', 'bg']
SP = sys.argv[1]
names = [r['display_name'] for r in csv.DictReader(open('data/yamnet/yamnet_class_map.csv', encoding='utf-8'))]
def idx(*keys): return [i for i, n in enumerate(names) if any(k.lower() in n.lower() for k in keys)]
GROUP = {'siren': idx('siren', 'emergency vehicle'),
         'car_horn': idx('vehicle horn', 'air horn', 'toot'),
         'crying_baby': idx('baby cry', 'crying, sobbing')}
K = int(os.environ.get('TOPK', 48))

def excess(img, ref, k=5): return float(np.max(np.sort(img - ref[:, None], axis=1)[:, -k:].mean(axis=1)))

class Split:
    def __init__(self, S): self.S = S; self.img = S['img']; self.sc = S['ysc']; self.em = S['yem']

def zs(sp, i):
    """YAMNet zero-shot: best of the three groups"""
    s = {t: float(sp.sc[i][GROUP[t]].max()) for t in EV}
    b = max(s, key=s.get); return b, s[b]

def codes(sp, i, gain_shift=None):
    e = sp.em[i]
    top = np.argpartition(-e, K)[:K]                       # most active embedding units
    cls = np.argpartition(-sp.sc[i], 8)[:8] + 1024         # most likely sound labels
    return np.concatenate([top, cls])
F = 1024 + 521

class Counting:
    """count tables over YAMNet's active units and top labels; learning = adding counts"""
    def __init__(self): self.c = {k: np.zeros(F) for k in CL}
    def add(self, k, cd): np.add.at(self.c[k], cd, 1)
    def copy(self):
        n = Counting(); n.c = {k: v.copy() for k, v in self.c.items()}; n.logp(); return n
    def logp(self): self.lp = {k: np.log((v + 0.5) / (v.sum() + 0.5 * F)) for k, v in self.c.items()}
    def scores(self, cd): return {k: float(self.lp[k][cd].mean()) for k in CL}

def evaluate(sp, decide, gain=None):
    """decide(i) -> (type or None). gain: use the mic-gain copy of the test set"""
    S = sp.S
    if gain is None:
        ev = [(t, s, i) for t in EV for s, i in S['testEv'][t]]; bg = S['testBg']; dis = S['testDis']
    else:
        G = S['testGain'][gain]; ev = G['ev']; bg = G['bg']; dis = G['dis']
    rec = {t: np.mean([decide(i) == t for tt, s, i in ev if tt == t]) for t in EV}
    bysnr = [np.mean([decide(i) == t for t, s, i in ev if s == q]) for q in SNRS]
    fab = np.mean([decide(i) is not None for i in bg]); fad = np.mean([decide(i) is not None for i in dis])
    out = dict(rec=rec, bysnr=bysnr, fab=fab, fad=fad)
    if gain is None:
        def ev_det(wins, t):
            h = [(decide(w) == t) if t else (decide(w) is not None) for w in wins]
            return any(sum(h[k:k + 3]) >= 2 for k in range(len(h) - 1))
        out['erec'] = {t: np.mean([ev_det(w, t) for s, w in S['clipEv'][t]]) for t in EV}
        out['esnr'] = [np.mean([ev_det(w, t) for t in EV for s, w in S['clipEv'][t] if s == q]) for q in SNRS]
        out['efab'] = np.mean([ev_det(w, None) for w in S['clipBg']]); out['efad'] = np.mean([ev_det(w, None) for w in S['clipDis']])
    return out

rows = {}
def put(k, v): rows.setdefault(k, []).append(v)
rolled = 0
for ti in range(4):
    S = pickle.load(open(os.path.join(SP, f'ysplit{ti}.pkl'), 'rb')); sp = Split(S)
    img = sp.img
    fref = np.mean([img[i].mean(axis=1) for i, _ in S['trainBg']], axis=0)
    cref = np.mean([img[i].mean(axis=1) for i in S['carStream']], axis=0)
    fdet = float(np.quantile([excess(img[i], fref) for i in S['calib'][:len(S['carStream'])]], 0.99))
    cdet = float(np.quantile([excess(img[i], cref) for i in S['carStream']], 0.99))
    # mic gain moves this car's learned background by the same amount (it is learned through the same mic)
    det_ok = lambda i, ref, det, g=0.: excess(img[i], ref + g) > det

    # --- Y0: YAMNet alone, thresholds so that 1% of training-car background/other windows trigger
    zc = [zs(sp, i) for i in S['calib']]
    zthr = {t: float(np.quantile([v for b, v in zc if b == t], 0.99)) if any(b == t for b, _ in zc) else 0.5 for t in EV}
    y0 = lambda i: (lambda b, v: b if v > zthr[b] else None)(*zs(sp, i))
    put('Y0 YAMNet만(범용 모델)', evaluate(sp, y0))
    put('Y0 + 마이크 +12dB', evaluate(sp, y0, 12.)); put('Y0 + 마이크 -12dB', evaluate(sp, y0, -12.))
    # --- Y1: + this car's background baseline detection (no labels)
    def y1f(g):
        return lambda i: (y0(i) if det_ok(i, cref, cdet, g) else None)
    put('Y1 + 이 차 배경 기준선 감지', evaluate(sp, y1f(0.)))
    put('Y1 + 마이크 +12dB', evaluate(sp, y1f(12.), 12.)); put('Y1 + 마이크 -12dB', evaluate(sp, y1f(-12.), -12.))

    # --- C1: counting on YAMNet outputs (factory data), with factory detection
    cm = Counting()
    for t in EV:
        for i, g in S['trainEv'][t]: cm.add(t, codes(sp, i))
    for i, g in S['trainOther']: cm.add('other', codes(sp, i))
    for i, g in S['trainBg']: cm.add('bg', codes(sp, i))
    cm.logp()
    def cdec(model, ref, det, thr, g=0.):
        def f(i):
            if not det_ok(i, ref, det, g): return None
            s = model.scores(codes(sp, i)); b = max(s, key=s.get)
            if b not in EV: return None
            return b if s[b] - max(s['other'], s['bg']) > thr.get(b, -9) else None
        return f
    def calib_thr(model, ref, det):
        ms = []
        for i in S['calib']:
            s = model.scores(codes(sp, i)); b = max(EV, key=lambda t: s[t]); ms.append((b, s[b] - max(s['other'], s['bg'])))
        return {t: float(np.quantile([m for b, m in ms if b == t], 0.95)) if sum(b == t for b, _ in ms) > 5 else -9 for t in EV}
    thr1 = calib_thr(cm, fref, fdet)
    put('C1 카운팅(YAMNet 출력 위, 출고)', evaluate(sp, cdec(cm, fref, fdet, thr1)))
    # --- C2: this car's background, learned on its own (counts + baseline + detection threshold)
    c2 = cm.copy()
    for i in S['carStream']:
        if cdec(c2, cref, cdet, thr1)(i) is None: c2.add('bg', codes(sp, i))
    c2.logp()
    put('C2 + 이 차 배경 자동 적응', evaluate(sp, cdec(c2, cref, cdet, thr1)))
    put('C2 + 마이크 +12dB', evaluate(sp, cdec(c2, cref, cdet, thr1, 12.), 12.)); put('C2 + 마이크 -12dB', evaluate(sp, cdec(c2, cref, cdet, thr1, -12.), -12.))
    # --- C3: events confirmed by the driver's reaction (first 9 per type); last 6 kept for the self test
    c3 = c2.copy()
    for t in EV:
        for i in S['confirmed'][t][:9]: c3.add(t, codes(sp, i))
    c3.logp()
    dec3 = cdec(c3, cref, cdet, thr1)
    put('C3 + 반응으로 확인된 사건음 학습', evaluate(sp, dec3))
    # --- C4: horns learned as sirens (twice the daily budget) -> self test on held-out confirmed sounds + car background
    def selftest(model):
        d = cdec(model, cref, cdet, thr1)
        rec = {t: sum(d(i) == t for i in S['confirmed'][t][9:]) for t in EV}
        fa = sum(d(i) is not None for i in S['carStream'][::4])
        return rec, fa
    good = selftest(c3)
    c4 = c3.copy()
    for _ in range(2):
        for i in S['wrong']: c4.add('siren', codes(sp, i))
    c4.logp()
    put('C4 잘못 학습(되돌림 없음)', evaluate(sp, cdec(c4, cref, cdet, thr1)))
    bad = selftest(c4)
    drop = any(bad[0][t] <= good[0][t] - 2 for t in EV) or bad[1] > good[1] + 2
    final = c3 if drop else c4
    rolled += int(drop)
    put('C4 잘못 학습 → 자가 시험 후', evaluate(sp, cdec(final, cref, cdet, thr1)))
    print('split', ti, S['target'], 'done', flush=True)

def avg(l):
    out = {}
    for k, v in l[0].items():
        if isinstance(v, dict): out[k] = {t: np.mean([x[k][t] for x in l]) for t in v}
        elif isinstance(v, list): out[k] = list(np.mean([x[k] for x in l], axis=0))
        else: out[k] = np.mean([x[k] for x in l])
    return out
print(f'\n=== 평균 (차 4대, 각 차는 학습에 쓰지 않고 평가) TOPK={K} ===')
for k, l in rows.items():
    a = avg(l)
    s = (f"{k:30s} 창: 사이렌 {a['rec']['siren']*100:3.0f}% 경적 {a['rec']['car_horn']*100:3.0f}% 울음 {a['rec']['crying_baby']*100:3.0f}% "
         f"(SNR +6/0/-6: {'/'.join(f'{x*100:.0f}' for x in a['bysnr'])}) 오경보 배경 {a['fab']*100:4.1f}% 다른소리 {a['fad']*100:4.1f}%")
    if 'erec' in a:
        s += (f" | 사건: 사이렌 {a['erec']['siren']*100:3.0f}% 경적 {a['erec']['car_horn']*100:3.0f}% 울음 {a['erec']['crying_baby']*100:3.0f}% "
              f"(SNR {'/'.join(f'{x*100:.0f}' for x in a['esnr'])}) 사건 오경보 배경 {a['efab']*100:4.1f}% 다른소리 {a['efad']*100:4.1f}%")
    print(s)
print('자가 시험 되돌림 발동:', rolled, '/ 4')
