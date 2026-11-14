# Tests 1-3 on the stored splits:
#  1) class statistics on YAMNet embeddings (shared-covariance LDA, streaming means; "other" and this car's background as negative classes)
#  2) temporal decision rules (2-of-3 vs hysteresis with per-class minimum duration)
#  3) siren frequency-track evidence
import sys, os, pickle, csv
import numpy as np
EV = ['siren', 'car_horn', 'crying_baby']
SNRS = [6., 0., -6.]
SP = sys.argv[1]
names = [r['display_name'] for r in csv.DictReader(open('data/yamnet/yamnet_class_map.csv', encoding='utf-8'))]
def idx(*keys): return [i for i, n in enumerate(names) if any(k.lower() in n.lower() for k in keys)]
GROUP = {'siren': idx('siren', 'emergency vehicle'), 'car_horn': idx('vehicle horn', 'air horn', 'toot'), 'crying_baby': idx('baby cry', 'crying, sobbing')}
def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
def excess(img, ref, k=5): return float(np.max(np.sort(img - ref[:, None], axis=1)[:, -k:].mean(axis=1)))
def auc(p, n):
    p = np.asarray(p, float); n = np.asarray(n, float); a = np.concatenate([p, n]); r = a.argsort().argsort() + 1
    return (r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))

class LDA:
    """class means + one shared covariance; means/covariance are running sums (no backprop)"""
    def __init__(self, d, lam=0.3): self.d = d; self.lam = lam; self.s = {}; self.n = {}; self.S = np.zeros((d, d)); self.N = 0
    def add(self, k, X):
        X = np.atleast_2d(X).astype(np.float64)
        self.s[k] = self.s.get(k, 0) + X.sum(0); self.n[k] = self.n.get(k, 0) + len(X)
        self.S += X.T @ X; self.N += len(X)
    def fit(self):
        mus = {k: self.s[k] / self.n[k] for k in self.s}
        tot = sum(self.s.values()) / self.N
        cov = self.S / self.N - np.outer(tot, tot)
        cov = (1 - self.lam) * cov + self.lam * np.trace(cov) / self.d * np.eye(self.d)
        P = np.linalg.inv(cov)
        self.W = {k: P @ m for k, m in mus.items()}; self.b = {k: -0.5 * m @ P @ m for k, m in mus.items()}
    def g(self, x): return {k: float(x @ self.W[k] + self.b[k]) for k in self.W}

def run(ti):
    S = pickle.load(open(os.path.join(SP, f'ysplit{ti}.pkl'), 'rb'))
    fm = pickle.load(open(os.path.join(SP, f'fm{ti}.pkl'), 'rb'))['fm']
    assert len(fm) == len(S['ysc'])
    img, sc, em = S['img'], S['ysc'], S['yem']
    fm = np.nan_to_num(fm)
    prof = {}
    for i, g in S['trainBg']: prof.setdefault(g, []).append(img[i].mean(axis=1))
    cref = np.mean([img[i].mean(axis=1) for i in S['carStream']], axis=0)
    exc = np.array([excess(img[i], cref) for i in S['carStream']]); mu_c, sd_c = exc.mean(), exc.std() + 1e-6
    # ---- previous best (F3): YAMNet group logit + car-baseline excess + counting correction of YAMNet labels
    L = lambda i: np.argpartition(-sc[i], 8)[:8]
    cev = {t: np.zeros(521) for t in EV}
    for t in EV:
        for i, g in S['trainEv'][t]: np.add.at(cev[t], L(i), 1)
    cbg = np.zeros(521)
    for i, g in S['trainBg'] + S['trainOther']: np.add.at(cbg, L(i), 1)
    for i in S['carStream']: np.add.at(cbg, L(i), 1)
    def f3(i):
        z = (excess(img[i], cref) - mu_c) / sd_c; l = L(i)
        return {t: float(logit(sc[i][GROUP[t]].max())) + z + float(np.mean(np.log((cev[t][l] + .5) / (cev[t].sum() + 260)) - np.log((cbg[l] + .5) / (cbg.sum() + 260)))) for t in EV}
    # ---- 1) LDA on embeddings: siren / horn / baby / other sounds / factory background / THIS car's background
    lda = LDA(1024)
    for t in EV: lda.add(t, em[[i for i, g in S['trainEv'][t]]])
    lda.add('other', em[[i for i, g in S['trainOther']]])
    lda.add('bg', em[[i for i, g in S['trainBg']]])
    lda.fit()
    def m1(model):
        def f(i):
            gg = model.g(em[i].astype(np.float64)); neg = max(v for k, v in gg.items() if k not in EV)
            return {t: gg[t] - neg for t in EV}
        return f
    lda_f = m1(lda)
    lda_c = LDA(1024); lda_c.s = dict(lda.s); lda_c.n = dict(lda.n); lda_c.S = lda.S.copy(); lda_c.N = lda.N
    lda_c.add('carbg', em[S['carStream']]); lda_c.fit()
    ldac_f = m1(lda_c)
    lda_r = LDA(1024); lda_r.s = dict(lda_c.s); lda_r.n = dict(lda_c.n); lda_r.S = lda_c.S.copy(); lda_r.N = lda_c.N
    for t in EV: lda_r.add(t, em[S['confirmed'][t][:9]])
    lda_r.fit(); ldar_f = m1(lda_r)
    # ---- 3) siren frequency-track evidence: small logistic on 7 track features (factory data: siren vs everything else)
    Xs = np.array([fm[i] for i, g in S['trainEv']['siren']]); Xn = np.array([fm[i] for t in ('car_horn', 'crying_baby') for i, g in S['trainEv'][t]] + [fm[i] for i, g in S['trainOther'] + S['trainBg']])
    X = np.vstack([Xs, Xn]); y = np.r_[np.ones(len(Xs)), np.zeros(len(Xn))]
    mu, sd = X.mean(0), X.std(0) + 1e-6; Xz = (X - mu) / sd; w = np.zeros(X.shape[1]); b = 0.
    cw = np.where(y == 1, len(y) / (2 * len(Xs)), len(y) / (2 * len(Xn)))
    for _ in range(500):
        p = 1 / (1 + np.exp(-(Xz @ w + b))); gr = (p - y) * cw; w -= 0.1 * (Xz.T @ gr / len(y) + 1e-3 * w); b -= 0.1 * gr.mean()
    fmz = lambda i: float(((fm[i] - mu) / sd) @ w + b)

    # per-car normalisation of each component (from this car's own unlabeled background) so they can be added
    def norm(f):
        v = np.array([[f(i)[t] for t in EV] for i in S['carStream']]); m, s = v.mean(0), v.std(0) + 1e-6
        return lambda i: (lambda d: {t: (d[t] - m[k]) / s[k] for k, t in enumerate(EV)})(f(i))
    nf3, nlda, nldac, nldar = norm(f3), norm(lda_f), norm(ldac_f), norm(ldar_f)
    fmv = np.array([fmz(i) for i in S['carStream']]); fm_m, fm_s = fmv.mean(), fmv.std() + 1e-6
    def plus(*fs, fmw=0.0):
        def f(i):
            out = {t: sum(g(i)[t] for g in fs) for t in EV}
            if fmw: out['siren'] += fmw * (fmz(i) - fm_m) / fm_s
            return out
        return f
    # ---- open-set check: is it close to stored examples of the event (kNN cosine), compared with other sounds / background?
    def l2(X): X = np.asarray(X, np.float64); return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-9)
    Ebank = {t: l2(em[[i for i, g in S['trainEv'][t]]]) for t in EV}
    Nbank = l2(em[[i for i, g in S['trainOther']] + [i for i, g in S['trainBg']] + list(S['carStream'])])
    KN = 5
    def knn(i):
        x = l2(em[i]); neg = np.sort(Nbank @ x)[-KN:].mean()
        return {t: float(np.sort(Ebank[t] @ x)[-KN:].mean() - neg) for t in EV}
    nknn = norm(knn)
    GAM = float(os.environ.get('GAM', 1.0))
    def gated(base, gam=GAM):
        def f(i):
            b = base(i); k = nknn(i); return {t: b[t] + gam * k[t] for t in EV}
        return f
    methods = {
        'F3 이전 최고(YAMNet+기준선+카운팅)': f3,
        'M1 임베딩 클래스 통계(출고)': lda_f,
        'M1 + 이 차 배경 프로토타입': ldac_f,
        'M1 + 반응 확인 사건음': ldar_f,
        'F3 + M1(이 차 배경)': plus(nf3, nldac),
        'F3 + M1 + 사이렌 궤적': plus(nf3, nldac, fmw=1.0),
        'F3 + M1(반응) + 사이렌 궤적': plus(nf3, nldar, fmw=1.0),
        'kNN 거리만': knn,
        '위 + kNN 미지 소리 거부 x1': gated(plus(nf3, nldar, fmw=1.0), 1.0),
        '위 + kNN 미지 소리 거부 x3': gated(plus(nf3, nldar, fmw=1.0), 3.0),
    }
    res = {}
    ev = [(t, s, i) for t in EV for s, i in S['testEv'][t]]
    for name, f in methods.items():
        any_ = lambda i: max(f(i).values())
        carv = np.array([any_(i) for i in S['carStream']])
        thr = float(np.quantile(carv, 0.98))                 # threshold from this car's own usual sound (2%)
        def dec(i):
            d = f(i); b = max(d, key=d.get); return b if d[b] > thr else None
        pos = [any_(i) for t, s, i in ev]; nb = [any_(i) for i in S['testBg']]; nd = [any_(i) for i in S['testDis']]
        sirpos = [f(i)['siren'] for s, i in S['testEv']['siren']]; sirneg = [f(i)['siren'] for i in S['testDis']]
        r = dict(aucb=auc(pos, nb), aucd=auc(pos, nd), aucsd=auc(sirpos, sirneg),
                 rec={t: np.mean([dec(i) == t for tt, s, i in ev if tt == t]) for t in EV},
                 fab=np.mean([dec(i) is not None for i in S['testBg']]), fad=np.mean([dec(i) is not None for i in S['testDis']]))
        # ---- 2) temporal rules on 5 s clips
        thr_off = float(np.quantile(carv, 0.90))
        MIN = {'siren': 2, 'car_horn': 1, 'crying_baby': 2}
        def rule23(wins, t):
            h = [(dec(w) == t) if t else (dec(w) is not None) for w in wins]
            return any(sum(h[k:k + 3]) >= 2 for k in range(len(h) - 1))
        def hyst(wins, t):
            ds = [f(w) for w in wins]; on = False; run_ = 0; votes = {}
            for d in ds:
                b = max(d, key=d.get); v = d[b]
                if not on and v > thr: on = True; run_ = 0; votes = {}
                if on:
                    if v > thr_off: run_ += 1; votes[b] = votes.get(b, 0) + 1
                    else:
                        on = False
                    lab = max(votes, key=votes.get) if votes else None
                    if lab and run_ >= MIN[lab] and (t is None or lab == t): return True
            return False
        for rn, rule in (('2of3', rule23), ('hyst', hyst)):
            r[rn] = dict(rec={t: np.mean([rule(w, t) for s, w in S['clipEv'][t]]) for t in EV},
                         snr=[np.mean([rule(w, t) for t in EV for s, w in S['clipEv'][t] if s == q]) for q in SNRS],
                         fab=np.mean([rule(w, None) for w in S['clipBg']]), fad=np.mean([rule(w, None) for w in S['clipDis']]))
        # trade-off: threshold so that other loud sounds trigger at most 10% / 20% of windows
        nd_ = np.array(nd)
        for cap in (0.1, 0.2):
            tc = float(np.quantile(nd_, 1 - cap))
            def decc(i, tc=tc):
                d = f(i); b = max(d, key=d.get); return b if d[b] > tc else None
            r[f'cap{cap}'] = dict(rec={t: np.mean([decc(i) == t for tt, s_, i in ev if tt == t]) for t in EV},
                                  fab=np.mean([decc(i) is not None for i in S['testBg']]))
        res[name] = r
    print('split', ti, S['target'], 'done', flush=True)
    return res

allr = [run(i) for i in range(4)]
m = lambda k, key: np.mean([r[k][key] for r in allr])
print('\n=== 창 단위 (차 4대 평균, 문턱 = 이 차 평소 소리의 상위 2%) ===')
for k in allr[0]:
    rec = {t: np.mean([r[k]['rec'][t] for r in allr]) for t in EV}
    print(f"{k:30s} AUC 배경 {m(k,'aucb'):.3f} 다른소리 {m(k,'aucd'):.3f} (사이렌 vs 다른소리 {m(k,'aucsd'):.3f}) | 재현율 사이렌 {rec['siren']*100:3.0f}% 경적 {rec['car_horn']*100:3.0f}% 울음 {rec['crying_baby']*100:3.0f}% | 오경보 배경 {m(k,'fab')*100:4.1f}% 다른소리 {m(k,'fad')*100:4.1f}%")
print('\n=== CAP: other loud sounds limited to 10% / 20% of windows ===')
for k in allr[0]:
    for cap in (0.1, 0.2):
        rec = {t: np.mean([r[k][f'cap{cap}']['rec'][t] for r in allr]) for t in EV}
        print(f"CAP {int(cap*100)}% | {k:30s} 사이렌 {rec['siren']*100:3.0f}% 경적 {rec['car_horn']*100:3.0f}% 울음 {rec['crying_baby']*100:3.0f}% | 배경 오경보 {np.mean([r[k][f'cap{cap}']['fab'] for r in allr])*100:4.1f}%")
print('\n=== 사건 단위 (5초 클립) ===')
for k in allr[0]:
    for rn in ('2of3', 'hyst'):
        rec = {t: np.mean([r[k][rn]['rec'][t] for r in allr]) for t in EV}; snr = np.mean([r[k][rn]['snr'] for r in allr], axis=0)
        print(f"{k:30s} {rn:4s} 사이렌 {rec['siren']*100:3.0f}% 경적 {rec['car_horn']*100:3.0f}% 울음 {rec['crying_baby']*100:3.0f}% (SNR +6/0/-6 {'/'.join(f'{x*100:.0f}' for x in snr)}) | 오경보 배경 {np.mean([r[k][rn]['fab'] for r in allr])*100:4.1f}% 다른소리 {np.mean([r[k][rn]['fad'] for r in allr])*100:4.1f}%")
