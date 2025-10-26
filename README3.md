```python
#does shape detection for an image

import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple

# Paths
MODEL_PATH   = "models/best_shape_clf.joblib"
LABEL_PATH   = "models/label_encoder.joblib"
DEFAULT_OUT  = "predictions.csv"
DEFAULT_PANEL_DIR = "panels"


FEATURES = ["area","perimeter","circularity","vertices","aspect_ratio","solidity"]

def _lazy_import_cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        raise ImportError("Install OpenCV: pip install opencv-python")

# Binarization
def _binarize_foreground(gray):
    cv2 = _lazy_import_cv2()
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th, 255 - th

# Find Contours
def _find_foreground_contours(gray, min_area_ratio=0.002, max_cover=0.95):
    cv2 = _lazy_import_cv2()
    H, W = gray.shape[:2]
    img_area = float(H * W)
    th, th_inv = _binarize_foreground(gray)

    def pick(binary):
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        good = []
        for c in cnts:
            a = cv2.contourArea(c)
            if a <= 1:
                continue
            r = a / img_area
            if r >= max_cover or r < min_area_ratio:
                continue
            good.append(c)
        return good

    allc = pick(th) + pick(th_inv)
    if not allc:
        return [], (H, W)

    def bbox(c):
        x, y, w, h = cv2.boundingRect(c)
        return (x, y, w, h)

    def biou(b1, b2):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        xa, ya = max(x1, x2), max(y1, y2)
        xb, yb = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        inter = max(0, xb - xa) * max(0, yb - ya)
        u = w1 * h1 + w2 * h2 - inter
        return inter / u if u > 0 else 0.0

    kept, boxes = [], []
    for c in sorted(allc, key=cv2.contourArea, reverse=True):
        b = bbox(c)
        if any(biou(b, bb) > 0.6 for bb in boxes):
            continue
        kept.append(c)
        boxes.append(b)

    return kept, (H, W)

# Split overlapping shapes
def _split_touching_in_region(gray, region_mask, min_area=25):
    cv2 = _lazy_import_cv2()
    fg = (region_mask > 0).astype(np.uint8)
    if fg.sum() == 0:
        return []
    dist = cv2.distanceTransform(fg, cv2.DIST_L2, 3)
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[(fg == 0)] = 0
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color, markers)

    contours = []
    for rid in range(2, markers.max() + 1):
        comp = (markers == rid).astype(np.uint8) * 255
        if comp.sum() < min_area:
            continue
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            contours.append(c)
    return contours

def _maybe_split_contour(gray, contour, split=True, solidity_thresh=0.93, max_vertices=12):
    if not split:
        return [contour]
    cv2 = _lazy_import_cv2()
    area = cv2.contourArea(contour)
    if area <= 1:
        return []
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull)) + 1e-9
    solidity = float(area / hull_area)
    peri = cv2.arcLength(contour, True)
    eps = 0.02 * peri if peri > 0 else 1.0
    approx = cv2.approxPolyDP(contour, eps, True)
    verts = len(approx)

    if solidity >= solidity_thresh and verts <= max_vertices:
        return [contour]

    x, y, w, h = cv2.boundingRect(contour)
    roi_mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(roi_mask, [contour - np.array([[x, y]])], -1, 255, -1)
    roi_gray = gray[y:y+h, x:x+w]
    split_cnts = _split_touching_in_region(roi_gray, roi_mask)
    if len(split_cnts) <= 1:
        return [contour]
    return [c + np.array([[x, y]]) for c in split_cnts]

# Feature extraction
def _features_from_contour(c):
    cv2 = _lazy_import_cv2()
    area = float(cv2.contourArea(c))
    peri = float(cv2.arcLength(c, True)) if area > 0 else 0.0
    circ = (4.0 * np.pi * area / (peri**2)) if peri > 0 else 0.0
    eps = 0.02 * peri if peri > 0 else 1.0
    approx = cv2.approxPolyDP(c, eps, True)
    vertices = int(len(approx))
    x, y, w, h = cv2.boundingRect(c)
    ar = (w / h) if h > 0 else 0.0
    hull = cv2.convexHull(c)
    hull_area = float(cv2.contourArea(hull))
    solidity = (area / hull_area) if hull_area > 0 else 0.0
    X = pd.DataFrame([[area, peri, circ, vertices, ar, solidity]], columns=FEATURES)
    bbox = (int(x), int(y), int(w), int(h))
    return X, bbox

# Detection pipeline
def _instances_from_image(image_path: str, do_split: bool):
    cv2 = _lazy_import_cv2()
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Could not read image: {image_path}")
    base_cnts, shape_hw = _find_foreground_contours(gray)
    final_cnts = []
    for c in base_cnts:
        final_cnts.extend(_maybe_split_contour(gray, c, split=do_split))
    feats, bboxes = [], []
    for c in final_cnts:
        X, bb = _features_from_contour(c)
        feats.append(X)
        bboxes.append(bb)
    return feats, final_cnts, bboxes, shape_hw

# Helpers
def align_features_for_model(X, model):
    feats = getattr(model, "feature_names_in_", None)
    if feats is None:
        return X, X
    feats = list(feats)
    X_pred = X[[c for c in feats if c in X.columns]].copy()
    X_pred = X_pred.reindex(columns=feats)
    return X, X_pred

def predict_with_confidence(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        y_idx = np.argmax(proba, axis=1)
        conf = proba[np.arange(len(X)), y_idx]
        return y_idx, conf, proba
    y_idx = model.predict(X)
    return y_idx, np.full(len(X), np.nan), None

# Rendering
def render_panel(image_path, contours, labels, confs, bboxes, save_path):
    cv2 = _lazy_import_cv2()
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return
    ann = img.copy()
    for i, c in enumerate(contours):
        cv2.drawContours(ann, [c], -1, (0, 255, 0), 2)
        x, y, w, h = bboxes[i]
        cv2.rectangle(ann, (x, y), (x+w, y+h), (255, 0, 0), 1)
        p = (x, max(0, y - 6))
        txt = f"{labels[i]} ({confs[i]:.2f})"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(ann, (p[0], p[1]-th-4), (p[0]+tw+6, p[1]+2), (0,0,0), -1)
        cv2.putText(ann, txt, (p[0]+3, p[1]-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), ann)

# Predict one image
def process_image(image_path, model, label_encoder, out_csv, panel_dir, split_touching):
    from collections import Counter
    feats, contours, bboxes, _ = _instances_from_image(image_path, do_split=split_touching)
    preds, confs, feat_rows = [], [], []
    for Xi, ci in zip(feats, contours):
        X_full, X_pred = align_features_for_model(Xi, model)
        y_idx, conf, proba = predict_with_confidence(model, X_pred)
        pred = label_encoder.inverse_transform(y_idx)[0]
        preds.append(str(pred))
        confs.append(float(conf[0]))
        feat_rows.append(X_full.iloc[0].to_dict())

    panel_path = Path(panel_dir) / (Path(image_path).stem + "_panel.png")
    render_panel(image_path, contours, preds, confs, bboxes, panel_path)

    rows = []
    for i in range(len(preds)):
        x, y, w, h = bboxes[i]
        row = {
            "filename": Path(image_path).name,
            "instance_id": i,
            "pred": preds[i],
            "confidence": confs[i],
            "bbox_x": x, "bbox_y": y, "bbox_w": w, "bbox_h": h,
        }
        row.update({f"feat_{k}": float(v) for k, v in feat_rows[i].items()})
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, mode="a", header=not Path(out_csv).exists(), index=False)

    counts = Counter(preds)
    print(f"\n{Path(image_path).name} — shape counts:")
    for lbl, n in sorted(counts.items()):
        print(f"  {lbl:>10s}: {n}")
    print(f"{image_path}: {len(preds)} shapes → {panel_path}")


def run_folder(folder, model, label_encoder, out_csv, panel_dir, split_touching):
    imgs = sorted([p for p in Path(folder).glob("*") if p.suffix.lower() in [".png",".jpg",".jpeg"]])
    if not imgs:
        print(f"No images in {folder}")
        return
    for img in imgs:
        try:
            process_image(str(img), model, label_encoder, out_csv, panel_dir, split_touching)
        except Exception as e:
            print(f"{img.name}: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", type=str, default="test_images", help="Folder with test images")
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--labels", default=LABEL_PATH)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--panel-dir", default=DEFAULT_PANEL_DIR)
    ap.add_argument("--split-touching", action="store_true", default=True)
    ap.add_argument("--no-split-touching", dest="split_touching", action="store_false")
    args = ap.parse_args()

    model = joblib.load(args.model)
    label_encoder = joblib.load(args.labels)

    run_folder(args.folder, model, label_encoder, args.out, args.panel_dir, args.split_touching)
    print(f"\nDone! Results saved to {args.out}")
    print(f"Annotated panels in: {args.panel_dir}")

if __name__ == "__main__":
    main()
