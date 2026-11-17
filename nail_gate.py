"""Decides whether a predicted mask is really a nail.

Two layers:
 1. Hard geometry rules (area, elongation, solidity)      - always on.
 2. Appearance/stability checks - either
      a) a logistic-regression model trained on YOUR good/bad examples (gate_lr.json,
         made by train_gate.py), or
      b) conservative default rules if gate_lr.json does not exist yet.
Every failure is returned as a human-readable reason, so a rejected photo means
"retake", never a colour verdict on the wrong pixels.
"""
import json
from pathlib import Path
import cv2, numpy as np

LR_PATH = Path(__file__).resolve().parent / "gate_lr.json"

# hard geometry rules
MIN_AREA, MAX_AREA = 0.005, 0.45
MAX_ELONGATION = 2.0
MIN_SOLIDITY = 0.85
# default appearance rules (used only when no trained LR is present)
MIN_CONFIDENCE, MIN_STABILITY, MIN_CONTRAST = 0.75, 0.80, 5.0
LR_ACCEPT = 0.5

FEATURES = ["area_ratio", "elongation", "solidity", "confidence", "stability",
            "contrast", "texture", "edge_touch"]


def _lab_mean(lab, m):
    return lab[m].mean(axis=0)


def features(bgr, mask, prob, stability):
    """Feature dict for one candidate mask. Works on the full-size image."""
    H, W = mask.shape[:2]
    m = mask > 0
    f = {k: 0.0 for k in FEATURES}
    if not m.any():
        return f
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    hull = cv2.contourArea(cv2.convexHull(c))
    f["area_ratio"] = float(m.mean())
    f["elongation"] = float(max(rw, rh) / max(min(rw, rh), 1e-6))
    f["solidity"] = float(cv2.contourArea(c) / hull) if hull > 0 else 0.0
    f["confidence"] = float(prob[m].mean())
    f["stability"] = float(stability)

    # colour contrast between the nail and the ring of pixels around it
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 100 / 255; lab[..., 1:] -= 128
    k = max(5, int(0.25 * np.sqrt(m.sum()))) | 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    outer = cv2.dilate(mask.astype(np.uint8), ker) > 0
    inner = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
    ring = outer & ~inner
    f["contrast"] = float(np.linalg.norm(_lab_mean(lab, m) - _lab_mean(lab, ring))) if ring.any() else 0.0

    # texture: nails are smooth, wood grain / skin creases are not.
    # Measured on a fixed-size (64 px) copy so it does not depend on photo resolution.
    ys, xs = np.where(m)
    crop = cv2.cvtColor(bgr[ys.min():ys.max()+1, xs.min():xs.max()+1], cv2.COLOR_BGR2GRAY)
    mc = m[ys.min():ys.max()+1, xs.min():xs.max()+1].astype(np.uint8)
    s = 64.0 / max(crop.shape)
    crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    mc = cv2.resize(mc, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    mc = cv2.erode(mc, np.ones((3, 3), np.uint8)) > 0
    if mc.sum() > 20:
        lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(crop, (0, 0), 1.0), cv2.CV_32F))
        f["texture"] = float(lap[mc].mean())

    # nails sit inside the finger; a mask hugging the photo edge is usually background
    e = max(2, int(0.01 * min(H, W)))
    border = np.zeros_like(m); border[:e] = border[-e:] = border[:, :e] = border[:, -e:] = True
    f["edge_touch"] = float((m & border).sum() / max(m.sum(), 1))
    return f


class NailGate:
    def __init__(self, lr_path=LR_PATH):
        self.lr = None
        p = Path(lr_path)
        if p.exists():
            self.lr = json.loads(p.read_text())

    def lr_prob(self, f):
        x = (np.array([f[k] for k in self.lr["features"]]) - np.array(self.lr["mean"])) / np.array(self.lr["scale"])
        z = float(np.dot(x, self.lr["coef"]) + self.lr["intercept"])
        return 1.0 / (1.0 + np.exp(-z))

    def check(self, bgr, res):
        """res = NailSegmenter.analyse() output. Returns dict(ok, reasons, features, lr_prob)."""
        mask = res["mask"]
        f = features(bgr, mask, res["prob"], res["stability"])
        reasons = []
        if not mask.any():
            return {"ok": False, "reasons": ["no nail found in the photo"], "features": f, "lr_prob": None}
        if not MIN_AREA <= f["area_ratio"] <= MAX_AREA:
            reasons.append("detected region is the wrong size for a nail")
        if f["elongation"] > MAX_ELONGATION:
            reasons.append("detected region is too long and thin to be a nail")
        if f["solidity"] < MIN_SOLIDITY:
            reasons.append("detected region has a ragged outline")

        p = None
        if self.lr is not None:
            p = self.lr_prob(f)
            if p < LR_ACCEPT:
                reasons.append(f"detected region does not look like a nail (score {p:.2f})")
        else:
            if f["confidence"] < MIN_CONFIDENCE:
                reasons.append("model is not confident this is a nail")
            if f["stability"] < MIN_STABILITY:
                reasons.append("detection is unstable between scans (cluttered background?)")
            if f["contrast"] < MIN_CONTRAST:
                reasons.append("detected region blends into its surroundings")
        return {"ok": not reasons, "reasons": reasons, "features": f, "lr_prob": p}
