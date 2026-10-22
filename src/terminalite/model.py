"""GRU copy sequence-to-sequence model (BiGRU encoder, attention, pointer-generator decoder) in numpy.

A re-implementation of the PyTorch training code: no torch, so start-up takes a fraction of a second.
On the same weights it returns the same commands up to float rounding.
"""

from pathlib import Path

import numpy as np

PAD, BOS, EOS = 0, 1, 2
MAX_SRC, MAX_TGT = 192, 96
LAYERS = 2


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


class GRUCopyModel:
    def __init__(self, path: Path) -> None:
        w = np.load(path)
        self.w = {k: w[k] for k in w.files}
        self.emb = self.w["emb.weight"]

    def _p(self, name: str) -> np.ndarray:
        return self.w[name]

    def _cell(self, gi: np.ndarray, h: np.ndarray, w_hh: np.ndarray, b_hh: np.ndarray) -> np.ndarray:
        """One GRU step given the input projection gi = x W_ih^T + b_ih (gates in torch's order r, z, n)."""
        gh = h @ w_hh.T + b_hh
        n_h = h.shape[1]
        r = sigmoid(gi[:, :n_h] + gh[:, :n_h])
        z = sigmoid(gi[:, n_h:2 * n_h] + gh[:, n_h:2 * n_h])
        n = np.tanh(gi[:, 2 * n_h:] + r * gh[:, 2 * n_h:])
        return (1 - z) * n + z * h

    def _enc_dir(self, x: np.ndarray, lens: np.ndarray, suffix: str, reverse: bool) -> np.ndarray:
        # Packed-sequence semantics: each row only sees its own length, padding outputs are zero.
        gi = x @ self._p(f"enc.weight_ih_{suffix}").T + self._p(f"enc.bias_ih_{suffix}")
        w_hh, b_hh = self._p(f"enc.weight_hh_{suffix}"), self._p(f"enc.bias_hh_{suffix}")
        b, s, _ = x.shape
        h = np.zeros((b, w_hh.shape[1]), np.float32)
        out = np.zeros((b, s, w_hh.shape[1]), np.float32)
        for t in (range(s - 1, -1, -1) if reverse else range(s)):
            valid = (t < lens)[:, None]
            h_new = self._cell(gi[:, t], h, w_hh, b_hh)
            h = np.where(valid, h_new, h)
            out[:, t] = np.where(valid, h_new, 0.0)
        return out

    def encode(self, src: list[list[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        lens = np.array([len(s) for s in src])
        ids = np.zeros((len(src), lens.max()), np.int64)
        for i, s in enumerate(src):
            ids[i, :len(s)] = s
        x = self.emb[ids]
        for layer in range(LAYERS):
            x = np.concatenate([self._enc_dir(x, lens, f"l{layer}", False),
                                self._enc_dir(x, lens, f"l{layer}_reverse", True)], -1)
        mask = ids != PAD
        # float32 throughout: one float64 operand makes every later matmul upcast the 16000x256 embedding.
        pooled = ((x * mask[..., None]).sum(1) / np.maximum(mask.sum(1, keepdims=True), 1)).astype(np.float32)
        return x, mask, pooled, ids

    def init_h(self, pooled: np.ndarray) -> np.ndarray:
        h = np.tanh(pooled @ self._p("experts.0.bridge.weight").T + self._p("experts.0.bridge.bias"))
        return h.reshape(len(pooled), LAYERS, -1).transpose(1, 0, 2).copy()

    def step(self, y: np.ndarray, h: np.ndarray, enc_out: np.ndarray, att_enc: np.ndarray, mask: np.ndarray,
             src_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Log-probabilities of the next token for previous tokens y (B,) and hidden state h (layers, B, H)."""
        e = self.emb[y]
        inp, h_new = e, []
        for layer in range(LAYERS):
            gi = inp @ self._p(f"experts.0.dec.weight_ih_l{layer}").T + self._p(f"experts.0.dec.bias_ih_l{layer}")
            inp = self._cell(gi, h[layer], self._p(f"experts.0.dec.weight_hh_l{layer}"), self._p(f"experts.0.dec.bias_hh_l{layer}"))
            h_new.append(inp)
        d = inp
        scores = (att_enc @ d[:, :, None])[:, :, 0]
        attn = softmax(np.where(mask, scores, -1e4))
        ctx = (attn[:, None, :] @ enc_out)[:, 0]
        dc = np.concatenate([d, ctx], -1)
        vocab_p = softmax(np.tanh(dc @ self._p("experts.0.comb.weight").T + self._p("experts.0.comb.bias")) @ self.emb.T)
        p_gen = sigmoid(np.concatenate([dc, e], -1) @ self._p("experts.0.gate.weight").T + self._p("experts.0.gate.bias"))
        # Copy distribution: attention mass summed per source token id (bincount is far faster than np.add.at).
        copy_p = np.stack([np.bincount(src_ids[i], weights=attn[i], minlength=vocab_p.shape[1]) for i in range(len(y))])
        copy_p = copy_p.astype(np.float32)
        return np.log(p_gen * vocab_p + (1 - p_gen) * copy_p + 1e-9), np.stack(h_new)

    def _context(self, src: list[list[int]]) -> tuple:
        enc_out, mask, pooled, ids = self.encode(src)
        return enc_out, enc_out @ self._p("experts.0.att.weight").T, mask, ids, self.init_h(pooled)

    def beam_nbest(self, src: list[int], k: int = 8, alpha: float = 0.6, max_len: int = MAX_TGT) -> list[tuple[list[int], float]]:
        """Finished hypotheses best first; scores are length-normalised log-probabilities (as gru_model.beam_nbest)."""
        enc_out, att_enc, mask, ids, h = self._context([src])
        beams = [([], 0.0, h[:, 0])]
        done: list[tuple[list[int], float]] = []
        for _ in range(max_len):
            n = len(beams)
            ys = np.array([b[0][-1] if b[0] else BOS for b in beams])
            hs = np.stack([b[2] for b in beams], 1)
            logp, h_new = self.step(ys, hs, np.repeat(enc_out, n, 0), np.repeat(att_enc, n, 0), np.repeat(mask, n, 0),
                                    np.repeat(ids, n, 0))
            part = np.argpartition(-logp, k, axis=1)[:, :k]
            top = np.take_along_axis(part, np.argsort(-np.take_along_axis(logp, part, 1), axis=1), 1)
            cand = [(seq + [int(t)], score + float(logp[i, t]), i) for i, (seq, score, _) in enumerate(beams) for t in top[i]]
            cand.sort(key=lambda c: -c[1])
            beams = []
            for seq, score, i in cand:
                if seq[-1] == EOS:
                    done.append((seq[:-1], score / len(seq) ** alpha))
                else:
                    beams.append((seq, score, h_new[:, i]))
                if len(beams) == k:
                    break
            if len(done) >= k or not beams:
                break
        if not done:
            done = [(b[0], b[1] / max(len(b[0]), 1) ** alpha) for b in beams]
        return sorted(done, key=lambda d: -d[1])

    def score(self, sources: list[list[int]], target: list[int]) -> np.ndarray:
        """Mean log-probability per token (EOS included) of one target under each source (as gru_model.score_pairs)."""
        enc_out, att_enc, mask, ids, h = self._context(sources)
        tgt = [BOS] + target[:MAX_TGT] + [EOS]
        total = np.zeros(len(sources))
        for t in range(len(tgt) - 1):
            logp, h = self.step(np.full(len(sources), tgt[t]), h, enc_out, att_enc, mask, ids)
            total += logp[:, tgt[t + 1]]
        return total / (len(tgt) - 1)
