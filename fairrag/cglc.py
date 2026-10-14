"""Confidence-Gated Logit Calibration (Sec. 3.3, Eqs. 5-11, App. A.6).

Binary one-vs-rest task per occupation o: the LLM scores 3 options (target + 2 distractors);
y_hat = 1  iff  the target option wins the argmax. Column 0 of `logits` is the target option.

Delta = (delta + delta_o) + (b + b_o) * s(g) + beta_{sign(z)} * z_o[g]        (Eq. 7 + App. A.6)
l_t'  = l_t - eta(l_t) * Delta,   eta = exp(-gamma * |l_t - mean(l)|)         (Eqs. 8-9)

Interpretation choices where the paper is under-specified are listed in README.md.
"""
from dataclasses import dataclass, field
import numpy as np
from .metrics import rd_one, tpr_gap_one


@dataclass
class Features:
    logits: np.ndarray      # (N,3) log-softmax over options, target first
    g: np.ndarray           # (N,) subject gender 0/1
    y: np.ndarray           # (N,) 1 if bio's profession == target occupation
    occ: np.ndarray         # (N,) int occupation id of the target
    S: np.ndarray           # (N,M) residual slate signal (signed NDKL, or skew proxy for ablation)

    def subset(self, idx):
        return Features(self.logits[idx], self.g[idx], self.y[idx], self.occ[idx], self.S[idx])


@dataclass
class Params:
    delta: float = 0.0
    gamma: float = 0.0
    b: float = 0.0
    beta_pos: float = 0.0
    beta_neg: float = 0.0
    delta_o: np.ndarray = None
    b_o: np.ndarray = None

    def copy(self):
        return Params(self.delta, self.gamma, self.b, self.beta_pos, self.beta_neg,
                      self.delta_o.copy(), self.b_o.copy())


class CGLC:
    def __init__(self, n_occ, use_gate=True, use_delta_o=True, use_b_o=True, use_residual=True,
                 clip=3.0, criterion="rd", acc_penalty=0.0, eps=1e-6, seed=42):
        self.n_occ, self.use_gate, self.use_delta_o, self.use_b_o = n_occ, use_gate, use_delta_o, use_b_o
        self.use_residual, self.clip, self.criterion = use_residual, clip, criterion
        self.acc_penalty, self.eps = acc_penalty, eps
        self.rng = np.random.default_rng(seed)
        self.params = Params(delta_o=np.zeros(n_occ), b_o=np.zeros(n_occ))
        self.mu = self.sd = None

    # ---- Eq. 5-6: per-class standardisation of the residual signal (training-set statistics)
    def fit_stats(self, f: Features):
        M = f.S.shape[1]
        self.mu = np.zeros((self.n_occ, M)); self.sd = np.ones((self.n_occ, M))
        for o in range(self.n_occ):
            m = (f.occ == o) & (f.y == 1)
            if m.sum() < 2:
                m = f.occ == o
            if m.sum() >= 2:
                self.mu[o], self.sd[o] = f.S[m].mean(0), f.S[m].std(0)

    def z(self, f: Features):
        mu, sd = self.mu[f.occ, f.g], self.sd[f.occ, f.g]
        return (f.S[np.arange(len(f.g)), f.g] - mu) / (sd + self.eps)

    # ---- Eq. 7-9
    def _pred(self, logits, g, occ, z, p: Params):
        sg = np.where(g == 1, 1.0, -1.0)
        beta = np.where(z >= 0, p.beta_pos, p.beta_neg)
        delta = p.delta + (p.delta_o[occ] if self.use_delta_o else 0.0)
        bb = p.b + (p.b_o[occ] if self.use_b_o else 0.0)
        D = delta + bb * sg + (beta * z if self.use_residual else 0.0)
        D = np.clip(D, -self.clip, self.clip)
        lt = logits[:, 0]
        eta = np.exp(-max(p.gamma, 0.0) * np.abs(lt - logits.mean(1))) if self.use_gate else 1.0
        lt_cal = lt - eta * D
        return lt_cal > logits[:, 1:].max(1)

    def predict(self, f: Features, p: Params = None, z=None):
        z = self.z(f) if z is None else z
        return self._pred(f.logits, f.g, f.occ, z, p or self.params)

    # ---- objective (Eq. 11): mean RD (or TPR gap for equal opportunity)
    def _gap(self, pred, g, y):
        return rd_one(pred, g) if self.criterion == "rd" else tpr_gap_one(pred, g, y)

    def _occ_obj(self, f, idx, z, p):
        pred = self._pred(f.logits[idx], f.g[idx], f.occ[idx], z[idx], p)
        obj = self._gap(pred, f.g[idx], f.y[idx])
        if self.acc_penalty:
            obj += self.acc_penalty * (1.0 - (pred == (f.y[idx] == 1)).mean())
        return obj

    def objective(self, f, z, idxs, p):
        return float(np.mean([self._occ_obj(f, ix, z, p) for ix in idxs]))

    # ---- App. A.6 search: A random -> B coordinate descent -> C per-class discrete
    def fit(self, f: Features, n_random=400, passes=4, class_grid=np.linspace(-0.5, 0.5, 11), verbose=True):
        self.fit_stats(f)
        z = self.z(f)
        idxs = [np.flatnonzero(f.occ == o) for o in range(self.n_occ)]
        idxs = [ix for ix in idxs if len(ix)]
        best = self.params.copy(); best_obj = self.objective(f, z, idxs, best)

        # Stage A: coarse random search over (delta, gamma, b, beta+, beta-)
        for _ in range(n_random):
            c = best.copy()
            c.delta, c.b = self.rng.uniform(-1, 1, 2)
            c.gamma = self.rng.uniform(0, 3) if self.use_gate else 0.0
            c.beta_pos, c.beta_neg = self.rng.uniform(-1, 1, 2)
            o = self.objective(f, z, idxs, c)
            if o < best_obj:
                best, best_obj = c, o
        if verbose: print(f"[CGLC] stage A obj={best_obj:.4f}")

        # Stage B: coordinate descent on the six globals (delta, gamma, b, beta+, beta-)
        steps = dict(delta=0.1, gamma=0.1, b=0.1, beta_pos=0.1, beta_neg=0.1)
        for _ in range(passes):
            improved = False
            for name, st in steps.items():
                if name == "gamma" and not self.use_gate:
                    continue
                for v in getattr(best, name) + st * np.arange(-5, 6):
                    c = best.copy(); setattr(c, name, max(v, 0.0) if name == "gamma" else v)
                    o = self.objective(f, z, idxs, c)
                    if o < best_obj - 1e-12:
                        best, best_obj, improved = c, o, True
            if not improved:
                break
        if verbose: print(f"[CGLC] stage B obj={best_obj:.4f}")

        # Stage C: per-class (delta_o, b_o) on a compact symmetric grid; classes are independent
        if self.use_delta_o or self.use_b_o:
            for _ in range(2):
                changed = False
                for o in range(self.n_occ):
                    ix = np.flatnonzero(f.occ == o)
                    if not len(ix):
                        continue
                    cur = self._occ_obj(f, ix, z, best)
                    for d in (class_grid if self.use_delta_o else [0.0]):
                        for bb in (class_grid if self.use_b_o else [0.0]):
                            c = best.copy(); c.delta_o[o], c.b_o[o] = d, bb
                            v = self._occ_obj(f, ix, z, c)
                            if v < cur - 1e-12:
                                best, cur, changed = c, v, True
                if not changed:
                    break
            best_obj = self.objective(f, z, idxs, best)
        if verbose: print(f"[CGLC] stage C obj={best_obj:.4f}")
        self.params = best
        return best_obj

    def state(self):
        p = self.params
        return dict(delta=p.delta, gamma=p.gamma, b=p.b, beta_pos=p.beta_pos, beta_neg=p.beta_neg,
                    delta_o=p.delta_o.tolist(), b_o=p.b_o.tolist(), mu=self.mu.tolist(), sd=self.sd.tolist())
