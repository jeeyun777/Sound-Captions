import sys, os, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proto import EV, SNRS
def auc(pos, neg):
    pos = np.asarray(pos); neg = np.asarray(neg)
    allv = np.concatenate([pos, neg]); r = allv.argsort().argsort() + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

SP = sys.argv[1]
for i in range(4):
    S = pickle.load(open(os.path.join(SP, f'split{i}.pkl'), 'rb'))
    lvl = lambda img: 10 * np.log10(np.mean(10 ** (img / 10)))
    nb = [lvl(x) for x in S['testBg']]
    row = []
    for q in SNRS:
        pos = [lvl(x) for t in EV for s, x in S['testEv'][t] if s == q]
        row.append(auc(pos, nb))
    # loudness relative to the car's own background, per band max
    ref = np.mean([x.mean(axis=1) for x in S['carStream']], axis=0)
    ex = lambda img: np.max(np.sort(img - ref[:, None], axis=1)[:, -5:].mean(axis=1))
    nb2 = [ex(x) for x in S['testBg']]
    row2 = [auc([ex(x) for t in EV for s, x in S['testEv'][t] if s == q], nb2) for q in SNRS]
    nd = [lvl(x) for x in S['testDis']]
    print(S['target'], 'overall level AUC vs bg by SNR', [round(v, 3) for v in row], '| band excess AUC', [round(v, 3) for v in row2],
          '| bg level mean %.1f sd %.2f' % (np.mean(nb), np.std(nb)))
