"""Train the logistic-regression nail gate on your own labelled photos.

    gate_data/good/   photos where the U-Net finds the nail correctly (include wood, cloth, etc.)
    gate_data/bad/    photos where it picks the wrong thing (wood strip, skin fold, ...)

    python train_gate.py            # writes gate_lr.json next to app.py
Aim for 30+ images per folder, with bad ones covering the backgrounds that fail today.
"""
import json, sys
from pathlib import Path
import cv2, numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "nailsegmenter_project"))
from nail_segmenter import NailSegmenter
import nail_gate as G


def collect(seg, folder, label):
    X, y, names = [], [], []
    for p in sorted(Path(folder).iterdir()):
        img = cv2.imread(str(p))
        if img is None:
            continue
        m = max(img.shape[:2])
        if m > 1600:
            img = cv2.resize(img, None, fx=1600 / m, fy=1600 / m, interpolation=cv2.INTER_AREA)
        r = seg.analyse(img)
        f = G.features(img, r["mask"], r["prob"], r["stability"])
        X.append([f[k] for k in G.FEATURES]); y.append(label); names.append(p.name)
    return X, y, names


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    data = BASE / "gate_data"
    seg = NailSegmenter()
    Xg, yg, _ = collect(seg, data / "good", 1)
    Xb, yb, _ = collect(seg, data / "bad", 0)
    X, y = np.array(Xg + Xb), np.array(yg + yb)
    print(f"{len(Xg)} good, {len(Xb)} bad")
    if len(Xg) < 10 or len(Xb) < 10:
        sys.exit("Need at least 10 images in each folder (30+ recommended).")
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    cv = min(5, len(Xg), len(Xb))
    print("cross-val accuracy: %.2f" % cross_val_score(clf, sc.transform(X), y, cv=cv).mean())
    clf.fit(sc.transform(X), y)
    for k, w in zip(G.FEATURES, clf.coef_[0]):
        print(f"  {k:12s} {w:+.2f}")
    G.LR_PATH.write_text(json.dumps({
        "features": G.FEATURES, "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
        "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0])}, indent=2))
    print("saved", G.LR_PATH)


if __name__ == "__main__":
    main()
