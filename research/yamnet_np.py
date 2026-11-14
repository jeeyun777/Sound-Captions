# YAMNet inference in numpy, following tensorflow/models research/audioset/yamnet (params.py, features.py, yamnet.py)
import numpy as np
import h5py

SR = 16000
WIN, HOP, NFFT = 400, 160, 512
MELS, FMIN, FMAX = 64, 125.0, 7500.0
PATCH = 96
EPS_BN = 1e-4
LAYERS = [('conv', 2, 32), ('sep', 1, 64), ('sep', 2, 128), ('sep', 1, 128), ('sep', 2, 256), ('sep', 1, 256),
          ('sep', 2, 512), ('sep', 1, 512), ('sep', 1, 512), ('sep', 1, 512), ('sep', 1, 512), ('sep', 1, 512),
          ('sep', 2, 1024), ('sep', 1, 1024)]

def _hz2mel(f): return 1127.0 * np.log(1.0 + f / 700.0)

def mel_matrix():
    # tf.signal.linear_to_mel_weight_matrix (HTK mel, DC bin dropped)
    nbins = NFFT // 2 + 1
    lin = np.linspace(0.0, SR / 2, nbins)[1:]
    lin_mel = _hz2mel(lin)[:, None]
    edges = np.linspace(_hz2mel(FMIN), _hz2mel(FMAX), MELS + 2)
    lo, ce, hi = edges[:-2][None], edges[1:-1][None], edges[2:][None]
    w = np.maximum(0.0, np.minimum((lin_mel - lo) / (ce - lo), (hi - lin_mel) / (hi - ce)))
    return np.vstack([np.zeros((1, MELS)), w]).astype(np.float32)

_MEL = mel_matrix()
_HANN = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN) / WIN)).astype(np.float32)   # periodic Hann

def log_mel(x):
    n = 1 + (len(x) - WIN) // HOP
    fr = np.stack([x[i * HOP:i * HOP + WIN] for i in range(n)]) * _HANN
    mag = np.abs(np.fft.rfft(fr, n=NFFT, axis=1)).astype(np.float32)
    return np.log(mag @ _MEL + 0.001)

class YAMNet:
    def __init__(self, path):
        f = h5py.File(path, 'r')
        g = lambda k: np.array(f[k], dtype=np.float32)
        self.L = []
        for i, (kind, stride, ch) in enumerate(LAYERS, 1):
            p = f'layer{i}'
            if kind == 'conv':
                self.L.append(('conv', stride, g(f'{p}/conv/{p}/conv/kernel:0'), self._bn(g, f'{p}/conv/bn/{p}/conv/bn')))
            else:
                self.L.append(('sep', stride,
                               g(f'{p}/depthwise_conv/{p}/depthwise_conv/depthwise_kernel:0')[:, :, :, 0],
                               self._bn(g, f'{p}/depthwise_conv/bn/{p}/depthwise_conv/bn'),
                               g(f'{p}/pointwise_conv/{p}/pointwise_conv/kernel:0')[0, 0],
                               self._bn(g, f'{p}/pointwise_conv/bn/{p}/pointwise_conv/bn')))
        self.Wl = g('logits/logits/kernel:0'); self.bl = g('logits/logits/bias:0')

    @staticmethod
    def _bn(g, p):
        beta, mean, var = g(f'{p}/beta:0'), g(f'{p}/moving_mean:0'), g(f'{p}/moving_variance:0')
        inv = 1.0 / np.sqrt(var + EPS_BN)
        return inv, beta - mean * inv          # y = x*inv + shift  (center=True, scale=False)

    @staticmethod
    def _pad(x, stride):
        # TF 'same' padding for a 3x3 kernel
        _, h, w, _ = x.shape
        def amt(n):
            out = -(-n // stride); tot = max((out - 1) * stride + 3 - n, 0); return tot // 2, tot - tot // 2, out
        a, b, oh = amt(h); c, d, ow = amt(w)
        return np.pad(x, ((0, 0), (a, b), (c, d), (0, 0))), oh, ow

    def _dw(self, x, k, stride):
        xp, oh, ow = self._pad(x, stride)
        out = np.zeros((x.shape[0], oh, ow, x.shape[3]), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                out += xp[:, i:i + stride * oh:stride, j:j + stride * ow:stride, :] * k[i, j]
        return out

    def _conv(self, x, k, stride):
        xp, oh, ow = self._pad(x, stride)
        out = np.zeros((x.shape[0], oh, ow, k.shape[3]), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                out += xp[:, i:i + stride * oh:stride, j:j + stride * ow:stride, :] @ k[i, j]
        return out

    def forward(self, patches):
        """patches: (B, 96, 64) log-mel -> (scores (B,521), embeddings (B,1024))"""
        x = patches[..., None].astype(np.float32)
        for layer in self.L:
            if layer[0] == 'conv':
                _, s, k, (inv, sh) = layer
                x = np.maximum(self._conv(x, k, s) * inv + sh, 0)
            else:
                _, s, dk, (i1, s1), pk, (i2, s2) = layer
                x = np.maximum(self._dw(x, dk, s) * i1 + s1, 0)
                x = np.maximum(x @ pk * i2 + s2, 0)
        emb = x.mean(axis=(1, 2))
        return 1 / (1 + np.exp(-(emb @ self.Wl + self.bl))), emb

    def run_windows(self, wins, batch=64):
        """wins: list of 1 s float waveforms (16 kHz) -> scores, embeddings (first 0.96 s patch)"""
        sc, em = [], []
        for i in range(0, len(wins), batch):
            p = np.stack([log_mel(w)[:PATCH] for w in wins[i:i + batch]])
            s, e = self.forward(p); sc.append(s); em.append(e)
        return np.concatenate(sc), np.concatenate(em)
