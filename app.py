import base64, sys
from pathlib import Path
import cv2, numpy as np
from flask import Flask, jsonify, render_template, request

BASE = Path(__file__).resolve().parent
SEG_DIR = BASE / "nailsegmenter_project"
sys.path.insert(0, str(SEG_DIR))
from nail_segmenter import NailSegmenter   # unchanged original module
from nail_gate import NailGate
import screening as S

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024
seg = NailSegmenter(weights=str(SEG_DIR / "best_unet.pth"))
gate = NailGate()   # uses gate_lr.json if present (see train_gate.py), else default rules

def b64(img_bgr, ext=".jpg"):
    ok, buf = cv2.imencode(ext, img_bgr)
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(buf).decode()

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/api/analyze")
def analyze():
    f = request.files.get("image")
    bgr = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR) if f else None
    if bgr is None:
        return jsonify(error="Please upload a valid image (JPG or PNG)."), 400
    m = max(bgr.shape[:2])
    if m > 1600:
        bgr = cv2.resize(bgr, None, fx=1600 / m, fy=1600 / m, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    q = S.quality_check(rgb)
    if not q["accepted"]:
        return jsonify(status="rejected", quality=q)

    res = seg.analyse(bgr)
    mask = res["mask"]
    g = gate.check(bgr, res)
    thick = max(2, bgr.shape[1] // 250)
    overlay = bgr.copy()
    if mask.any():
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (0, 200, 0) if g["ok"] else (0, 0, 220), thick)

    if not g["ok"]:
        # Never analyse colour on a region that is not verified as a nail.
        return jsonify(status="retake", quality=q, reasons=g["reasons"],
                       overlay=b64(overlay) if mask.any() else None)

    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    roi, nm = rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1].astype(bool)
    crop = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR) * nm[:, :, None].astype(np.uint8)
    cos = S.cosmetic_check(roi, nm)
    feats = S.extract_features(roi, nm)
    flags = [] if cos["suspected"] else S.screen(feats)
    return jsonify(status="ok", quality=q, method="AI segmentation (verified)", nail_found=True,
                   cosmetic=cos, features=feats, flags=flags,
                   overlay=b64(overlay), crop=b64(crop, ".png"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
