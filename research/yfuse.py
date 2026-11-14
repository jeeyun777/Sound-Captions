# Fusion: YAMNet evidence + "how far above this car's usual sound" + counting correction learned in this car.
import sys, os, pickle, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
EV = ['siren', 'car_horn', 'crying_baby']
SNRS = [6., 0., -6.]
SP = sys.argv[1]
FA = float(os.environ.get('FA', 0.01))
names = [r['display_name'] for r in csv.DictReader(open('data/yamnet/yamnet_class_map.csv', encoding='utf-8'))]
def idx(*keys): return [i for i, n in enumerate(names) if any(k.lower() in n.lower() for k in keys)]
GROUP = {'siren': idx('siren', 'emergency vehicle'), 'car_horn': idx('vehicle horn', 'air horn', 'toot'), 'crying_baby': idx('baby cry', 'crying, sobbing')}
TOPL = 8

def excess(img, ref, k=5): return float(np.max(np.sort(img - ref[:, None], axis=1)[:, -k:].mean(axis=1)))
def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))

def run(ti):
    S = pickle.load(open(os.path.join(SP, f'ysplit{ti}.pkl'), 'rb'))
    img, sc = S['img'], S['ysc']
    # per-car background profiles (training cars from their own background, target car from its own stream)
    prof = {}
    for i, g in S['trainBg']: prof.setdefault(g, []).append(img[i].mean(axis=1))
    prof = {g: np.mean(v, axis=0) for g, v in prof.items()}
    fref = np.mean(list(prof.values()), axis=0)
    cref = np.mean([img[i].mean(axis=1) for i in S['carStream']], axis=0)
    ex_car = np.array([excess(img[i], cref) for i in S['carStream']]); mu_c, sd_c = ex_car.mean(), ex_car.std() + 1e-6
    calib_ex = np.array([excess(img[i], fref) for i in S['calib']]); mu_f, sd_f = calib_ex[:len(ex_car)].mean(), calib_ex[:len(ex_car)].std() + 1e-6
    def y(i): return {t: float(logit(sc[i][GROUP[t]].max())) for t in EV}
    def labels(i): return np.argpartition(-sc[i], TOPL)[:TOPL]

    # counting tables over YAMNet's top labels: events (factory data) vs background (factory, then this car)
    cnt_ev = {t: np.zeros(521) for t in EV}
    for t in EV:
        for i, g in S['trainEv'][t]: np.add.at(cnt_ev[t], labels(i), 1)
    cnt_bg_f = np.zeros(521)
    for i, g in S['trainBg']: np.add.at(cnt_bg_f, labels(i), 1)
    for i, g in S['trainOther']: np.add.at(cnt_bg_f, labels(i), 1)
    def llr(i, cev, cbg):
        L = labels(i)
        return {t: float(np.mean(np.log((cev[t][L] + 0.5) / (cev[t].sum() + 260)) - np.log((cbg[L] + 0.5) / (cbg.sum() + 260)))) for t in EV}

    def make(use_ex, ref, mu, sd, cbg=None, cev=None, w_ex=1.0, w_c=1.0):
        def score(i, gain=0.):
            s = y(i)
            if use_ex:
                z = (excess(img[i], ref + gain) - mu) / sd
                s = {t: s[t] + w_ex * z for t in EV}
            if cbg is not None:
                l = llr(i, cev, cbg); s = {t: s[t] + w_c * l[t] for t in EV}
            return s
        return score
    def calibrate(score, pool, gain=0.):
        ms = [score(i, gain) for i in pool]
        return {t: float(np.quantile([m[t] for m in ms], 1 - FA)) for t in EV}
    def decider(score, thr, gain=0.):
        def d(i):
            s = score(i, gain); c = {t: s[t] - thr[t] for t in EV}; b = max(c, key=c.get)
            return b if c[b] > 0 else None
        return d

    res = {}
    # Y0: YAMNet alone, thresholds from training cars (background + other sounds)
    s0 = make(False, None, 0, 1); th0 = calibrate(s0, S['calib'])
    res['Y0 YAMNet만(범용 모델)'] = (s0, th0, False)
    # F1: + how far above the (factory) background baseline
    s1 = make(True, fref, mu_f, sd_f); th1 = calibrate(s1, S['calib'])
    res['F1 + 배경 기준선 초과량(출고 기준선)'] = (s1, th1, False)
    # F2: this car: own baseline, thresholds from this car's own usual sound (no labels) -- never below factory
    s2 = make(True, cref, mu_c, sd_c)
    th2c = calibrate(s2, S['carStream']); th2 = {t: max(th2c[t], calibrate(s2, S['calib'])[t] - 1.0) for t in EV}
    res['F2 + 이 차 기준선·문턱 자동 적응'] = (s2, th2, True)
    # F3: + counting correction: which YAMNet labels this car's own background keeps producing
    cbg = cnt_bg_f.copy()
    for i in S['carStream']: np.add.at(cbg, labels(i), 1)
    s3 = make(True, cref, mu_c, sd_c, cbg, cnt_ev)
    th3c = calibrate(s3, S['carStream']); th3 = {t: max(th3c[t], calibrate(s3, S['calib'])[t] - 1.0) for t in EV}
    res['F3 + 카운팅 보정(이 차 배경 라벨)'] = (s3, th3, True)
    # F4: + events confirmed by the driver's reaction
    cev = {t: cnt_ev[t].copy() for t in EV}
    for t in EV:
        for i in S['confirmed'][t][:9]: np.add.at(cev[t], labels(i), 1)
    s4 = make(True, cref, mu_c, sd_c, cbg, cev)
    th4c = calibrate(s4, S['carStream']); th4 = {t: max(th4c[t], calibrate(s4, S['calib'])[t] - 1.0) for t in EV}
    res['F4 + 반응으로 확인된 사건음 학습'] = (s4, th4, True)
    # F5: + YAMNet knows what else it is: if a clearly different sound (not background-like) wins, push down
    BGLIKE = set(idx('vehicle', 'engine', 'car', 'rain', 'wind', 'train', 'aircraft', 'helicopter', 'noise', 'rumble', 'water', 'motor', 'traffic', 'silence', 'inside, small room', 'outside', 'hum', 'mechanisms', 'speech'))
    NON = np.array([k for k in range(521) if k not in set(sum(GROUP.values(), [])) and k not in BGLIKE])
    BETA = float(os.environ.get('BETA', 0.7))
    def s5(i, gain=0.):
        b = s4(i, gain); o = float(logit(sc[i][NON].max())); yy = y(i)
        return {t: b[t] - BETA * max(0.0, o - yy[t]) for t in EV}
    th5c = calibrate(s5, S['carStream']); th5 = {t: max(th5c[t], calibrate(s5, S['calib'])[t] - 1.0) for t in EV}
    res['F5 + 다른 소리 억제(YAMNet 라벨)'] = (s5, th5, True)
    def s5y(i, gain=0.):
        b = s0(i, gain); o = float(logit(sc[i][NON].max())); yy = y(i)
        return {t: b[t] - BETA * max(0.0, o - yy[t]) for t in EV}
    res['Y0b YAMNet + 다른 소리 억제만'] = (s5y, calibrate(s5y, S['calib']), False)

    out = {}
    for name, (score, thr, adapted) in res.items():
        d = decider(score, thr)
        ev = [(t, s, i) for t in EV for s, i in S['testEv'][t]]
        r = dict(rec={t: np.mean([d(i) == t for tt, s, i in ev if tt == t]) for t in EV},
                 bysnr=[np.mean([d(i) == t for t, s, i in ev if s == q]) for q in SNRS],
                 fab=np.mean([d(i) is not None for i in S['testBg']]), fad=np.mean([d(i) is not None for i in S['testDis']]))
        def ev_det(wins, t):
            h = [(d(w) == t) if t else (d(w) is not None) for w in wins]
            return any(sum(h[k:k + 3]) >= 2 for k in range(len(h) - 1))
        r['erec'] = {t: np.mean([ev_det(w, t) for s, w in S['clipEv'][t]]) for t in EV}
        r['esnr'] = [np.mean([ev_det(w, t) for t in EV for s, w in S['clipEv'][t] if s == q]) for q in SNRS]
        r['efab'] = np.mean([ev_det(w, None) for w in S['clipBg']]); r['efad'] = np.mean([ev_det(w, None) for w in S['clipDis']])
        def anys(i): sc_ = score(i); return max(sc_[t] - thr[t] for t in EV)
        pos = [anys(i) for t, s_, i in ev]; pos6 = [anys(i) for t, s_, i in ev if s_ >= 0]
        nb = [anys(i) for i in S['testBg']]; nd = [anys(i) for i in S['testDis']]
        def auc(p_, n_):
            p_ = np.asarray(p_); n_ = np.asarray(n_); a = np.concatenate([p_, n_]); rk = a.argsort().argsort() + 1
            return (rk[:len(p_)].sum() - len(p_) * (len(p_) + 1) / 2) / (len(p_) * len(n_))
        r['aucb'] = auc(pos, nb); r['aucd'] = auc(pos, nd); r['aucb0'] = auc(pos6, nb); r['aucd0'] = auc(pos6, nd)
        # recall at 5% false alarms on this test car's background (same operating point for every method)
        cut = float(np.quantile(nb, 0.95)); r['rec5'] = float(np.mean(np.array(pos) > cut))
        out[name] = r
        for g in (12., -12.):
            G = S['testGain'][g]; dg = decider(score, thr, g if adapted or 'F1' in name else 0.)
            out[f'{name[:2]} 마이크 {g:+.0f}dB'] = dict(rec={t: np.mean([dg(i) == t for tt, s, i in G['ev'] if tt == t]) for t in EV},
                bysnr=[np.mean([dg(i) == t for t, s, i in G['ev'] if s == q]) for q in SNRS],
                fab=np.mean([dg(i) is not None for i in G['bg']]), fad=np.mean([dg(i) is not None for i in G['dis']]))
    print('split', ti, S['target'], 'done', flush=True)
    return out

allr = [run(i) for i in range(4)]
print(f'\n=== 평균 (차 4대, 학습에 쓰지 않은 차로 평가, 목표 오경보 {FA*100:.0f}%) ===')
for k in allr[0]:
    l = [r[k] for r in allr]
    rec = {t: np.mean([x['rec'][t] for x in l]) for t in EV}; snr = np.mean([x['bysnr'] for x in l], axis=0)
    s = (f"{k:30s} 창: 사이렌 {rec['siren']*100:3.0f}% 경적 {rec['car_horn']*100:3.0f}% 울음 {rec['crying_baby']*100:3.0f}% "
         f"(SNR +6/0/-6: {'/'.join(f'{x*100:.0f}' for x in snr)}) 오경보 배경 {np.mean([x['fab'] for x in l])*100:4.1f}% 다른소리 {np.mean([x['fad'] for x in l])*100:4.1f}%")
    if 'erec' in l[0]:
        er = {t: np.mean([x['erec'][t] for x in l]) for t in EV}; es = np.mean([x['esnr'] for x in l], axis=0)
        s += (f" | 사건: 사이렌 {er['siren']*100:3.0f}% 경적 {er['car_horn']*100:3.0f}% 울음 {er['crying_baby']*100:3.0f}% "
              f"(SNR {'/'.join(f'{x*100:.0f}' for x in es)}) 오경보 배경 {np.mean([x['efab'] for x in l])*100:4.1f}% 다른소리 {np.mean([x['efad'] for x in l])*100:4.1f}%")
    if 'aucb' in l[0]:
        s += " | AUC 배경 %.3f(0dB↑ %.3f) 다른소리 %.3f(0dB↑ %.3f) 오경보5%%때 재현율 %.0f%%" % tuple(
            np.mean([x[k2] for x in l]) * (100 if k2 == 'rec5' else 1) for k2 in ('aucb', 'aucb0', 'aucd', 'aucd0', 'rec5'))
    print(s)
