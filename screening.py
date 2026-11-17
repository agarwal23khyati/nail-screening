"""Screening logic ported from Another_copy_of_nail_screening_v3_result.ipynb (thresholds unchanged)."""
import cv2, numpy as np
from scipy import ndimage as ndi
from skimage.color import rgb2lab
from skimage.filters import threshold_multiotsu
from skimage.measure import label, regionprops

BLUR_MIN, GLARE_MAX_FRAC, DARK_MIN = 3.0, 0.02, 60
SAT_MAX, DARK_NAIL_MAX = 140, 70
PALE_L_MIN, PALE_A_MAX, YELLOW_B_MIN, BLUE_B_MAX = 78, 8, 25, 0

def quality_check(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    glare = float(((hsv[:, :, 2] > 240) & (hsv[:, :, 1] < 40)).mean())
    bright = float(gray.mean())
    problems = []
    if blur < BLUR_MIN: problems.append("Image is blurry - hold steady and focus on the nail.")
    if glare > GLARE_MAX_FRAC: problems.append("Glare detected - avoid flash and direct tubelight.")
    if bright < DARK_MIN: problems.append("Image is too dark - use soft natural light.")
    return {"accepted": not problems, "problems": problems,
            "blur": blur, "glare_pct": glare * 100, "brightness": bright}

def get_nail_roi(img_rgb):
    """Colour-based fallback. Returns (roi, mask, box, found)."""
    h, w = img_rgb.shape[:2]
    s = 300.0 / max(h, w)
    small = cv2.resize(img_rgb, (max(1, int(w*s)), max(1, int(h*s))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    L = cv2.GaussianBlur(rgb2lab(small / 255.0)[..., 0].astype(np.float32), (0, 0), 2)
    best, best_score = None, 0
    try:
        t = threshold_multiotsu(L, classes=3)
        class_masks = [L > t[1], (L > t[0]) & (L <= t[1])]
    except Exception:
        class_masks = []
    for m in class_masks:
        m = ndi.binary_fill_holes(ndi.binary_opening(m, iterations=2))
        lab = label(m)
        for r in regionprops(lab):
            af = r.area / (sh * sw)
            if af < 0.02 or af > 0.5: continue
            minr, minc, maxr, maxc = r.bbox
            touches = int(minr == 0) + int(minc == 0) + int(maxr == sh) + int(maxc == sw)
            if touches >= 2: continue
            cy, cx = r.centroid
            cen = 1 - np.hypot(cy/sh - 0.5, cx/sw - 0.5) / 0.71
            score = np.sqrt(af) * cen * r.solidity * (1 - 0.3 * touches)
            if score > best_score: best_score, best = score, (r, lab == r.label)
    if best is None:
        ch, cw = int(h*0.4), int(w*0.4); y0, x0 = (h-ch)//2, (w-cw)//2
        roi = img_rgb[y0:y0+ch, x0:x0+cw]
        return roi, np.ones(roi.shape[:2], bool), (y0, y0+ch, x0, x0+cw), False
    r, ms = best
    it = max(2, int(0.12 * r.equivalent_diameter))
    er = ndi.binary_erosion(ms, iterations=it)
    if er.sum() < 0.3 * ms.sum(): er = ms
    full = cv2.resize(er.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    ys, xs = np.where(full)
    y0, y1, x0, x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
    return img_rgb[y0:y1, x0:x1], full[y0:y1, x0:x1], (y0, y1, x0, x1), True

def cosmetic_check(roi_rgb, mask=None):
    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    px = hsv[mask] if mask is not None else hsv.reshape(-1, 3)
    s, v, hue = px[:, 1].mean(), px[:, 2].mean(), np.median(px[:, 0].astype(float) * 2)
    reasons = []
    if s > SAT_MAX: reasons.append("high colour saturation")
    if v < DARK_NAIL_MAX: reasons.append("very dark nail area")
    if 90 < hue < 300: reasons.append("unusual hue for a bare nail")
    return {"suspected": bool(reasons), "reasons": reasons}

def extract_features(roi_rgb, mask=None):
    px = (roi_rgb[mask] if mask is not None else roi_rgb.reshape(-1, 3)).astype(np.float32)
    R, G, B = px[:, 0].mean(), px[:, 1].mean(), px[:, 2].mean()
    hsv = cv2.cvtColor(px.astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    lab = rgb2lab((px / 255.0).reshape(-1, 1, 3)).reshape(-1, 3)
    f = {"R": R, "G": G, "B": B, "H": hsv[:, 0].mean()*2, "S": hsv[:, 1].mean(), "V": hsv[:, 2].mean(),
         "L": lab[:, 0].mean(), "a": lab[:, 1].mean(), "b": lab[:, 2].mean(), "L_std": lab[:, 0].std()}
    return {k: float(v) for k, v in f.items()}

def screen(f):
    flags = []
    if f["L"] > PALE_L_MIN and f["a"] < PALE_A_MAX and f["b"] < YELLOW_B_MIN:
        flags.append({"title": "Pale nails", "meaning": "Possible anemia / low hemoglobin, malnutrition or liver issue",
                      "tests": "Hemoglobin (CBC), iron studies, liver function"})
    if f["b"] > YELLOW_B_MIN:
        flags.append({"title": "Yellowish nails", "meaning": "Possible diabetes, high cholesterol, liver/jaundice, fungal infection or smoking stain",
                      "tests": "Fasting sugar / HbA1c, lipid profile, liver function"})
    if f["b"] < BLUE_B_MAX and f["a"] < 15:
        flags.append({"title": "Bluish tint", "meaning": "Possible low oxygen (heart/lung issue)",
                      "tests": "See a doctor, pulse oximetry"})
    return flags
