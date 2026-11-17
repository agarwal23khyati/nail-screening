# Nail Screening - run

    pip install -r requirements.txt
    python app.py

Open http://localhost:5000

## How non-uniform backgrounds (wood, cloth, ...) are handled
- The old colour-based fallback is removed. If a nail cannot be verified, the app says "Retake needed" and
  gives NO colour verdict.
- U-Net runs with multi-scale + flip test-time augmentation; unstable detections are rejected.
- `nail_gate.py` checks geometry, confidence, stability, contrast with surroundings and texture.
- Optional but recommended: train the logistic-regression gate on your own photos:
      gate_data/good/   (nail found correctly, include wood-background photos)
      gate_data/bad/    (wrong region picked)
      python train_gate.py        # writes gate_lr.json; app.py picks it up on restart
