from pathlib import Path
import cv2, numpy as np, torch
import segmentation_models_pytorch as smp

PROB_THRESH = 0.5


class NailSegmenter:
    SIZE = 512
    TTA_SIZES = (384, 512, 640)      # multi-scale test-time augmentation (multiples of 32)
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, weights=Path(__file__).resolve().parent / "best_unet.pth", device=None):
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = smp.Unet("mobilenet_v2", encoder_weights=None,
                              in_channels=3, classes=1).to(self.dev)
        self.model.load_state_dict(torch.load(str(weights), map_location=self.dev))
        self.model.eval()

    # ---------- inference ----------
    def _prob(self, bgr, size, flip=False):
        h, w = bgr.shape[:2]
        x = cv2.cvtColor(cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        if flip:
            x = x[:, ::-1]
        x = ((x.astype(np.float32) / 255 - self.MEAN) / self.STD).transpose(2, 0, 1)
        with torch.no_grad():
            p = torch.sigmoid(self.model(torch.from_numpy(np.ascontiguousarray(x))[None].to(self.dev)))[0, 0].cpu().numpy()
        if flip:
            p = p[:, ::-1]
        return cv2.resize(np.ascontiguousarray(p), (w, h), interpolation=cv2.INTER_LINEAR)

    def predict_prob(self, bgr, tta=True):
        """Returns (mean_prob, [individual_probs]). Real nails give stable predictions
        across scales/flips; hallucinated masks on textured backgrounds usually do not."""
        runs = [(self.SIZE, False)]
        if tta:
            runs = [(s, f) for s in self.TTA_SIZES for f in (False, True)]
        probs = [self._prob(bgr, s, f) for s, f in runs]
        return np.mean(probs, axis=0), probs

    # ---------- mask clean-up ----------
    @staticmethod
    def clean_mask(prob):
        """Threshold; keep the component with best (area x confidence); smooth; fill holes."""
        mask = (prob > PROB_THRESH).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
        if n <= 1:
            return mask
        best, best_score = 1, -1.0
        for i in range(1, n):
            score = st[i, cv2.CC_STAT_AREA] * float(prob[lab == i].mean())
            if score > best_score:
                best, best_score = i, score
        mask = (lab == best).astype(np.uint8)
        k = max(3, int(round(0.03 * np.sqrt(mask.sum()))) | 1)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = np.zeros_like(mask)
        if cnts:
            cv2.drawContours(out, cnts, -1, 1, thickness=cv2.FILLED)
        return out

    @staticmethod
    def stability(mask, probs):
        """Mean IoU between the final mask and each TTA run's own mask (1.0 = fully stable)."""
        m = mask > 0
        ious = []
        for p in probs:
            q = p > PROB_THRESH
            u = (m | q).sum()
            ious.append((m & q).sum() / u if u else 0.0)
        return float(np.mean(ious))

    # ---------- main entry ----------
    def predict_mask(self, bgr):
        """Backward-compatible: cleaned binary mask only."""
        prob, _ = self.predict_prob(bgr, tta=False)
        return self.clean_mask(prob)

    def analyse(self, bgr, tta=True):
        """Returns dict(mask, prob, probs, stability). Validation lives in nail_gate.py."""
        prob, probs = self.predict_prob(bgr, tta=tta)
        mask = self.clean_mask(prob)
        return {"mask": mask, "prob": prob, "probs": probs,
                "stability": self.stability(mask, probs) if mask.any() else 0.0}

    @staticmethod
    def cut(bgr, mask):
        ys, xs = np.where(mask)
        return (bgr * mask[:, :, None])[ys.min():ys.max()+1, xs.min():xs.max()+1]

    def crop(self, bgr):
        """Returns (crop_or_None, mask, ok). Uses the strict gate when available."""
        r = self.analyse(bgr, tta=False)
        mask = r["mask"]
        if mask.sum() == 0:
            return None, mask, False
        try:
            from nail_gate import NailGate
            ok = NailGate().check(bgr, r)["ok"]
        except ImportError:
            ok = True
        return (self.cut(bgr, mask) if ok else None), mask, ok


if __name__ == "__main__":
    import sys
    seg = NailSegmenter()
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "output")
    out.mkdir(parents=True, exist_ok=True)
    p = Path(sys.argv[1])
    crop, mask, ok = seg.crop(cv2.imread(str(p)))
    print("valid:", ok)
    if crop is not None:
        cv2.imwrite(str(out / f"{p.stem}_nail.png"), crop)
