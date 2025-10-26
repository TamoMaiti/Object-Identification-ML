```python
#generates data for training
import os, csv, math, random
from math import cos, sin, pi
from pathlib import Path
import numpy as np
import cv2

# Configuration
OUT_DIR = "ml_one_shape_per_image"
IMG_SIZE = 256
NUM_IMAGES = 300
SHAPES = ["circle", "triangle", "square", "pentagon", "hexagon", "star5"]
WEIGHTS = [1.0, 1.0, 1.0, 1.2, 1.2, 1.0] 

# Transform settings
SCALE_MIN, SCALE_MAX = 0.6, 1.4
ROT_MIN_DEG, ROT_MAX_DEG = -45.0, 45.0
BASE_SIZE_MIN, BASE_SIZE_MAX = 28, 64

# Noise settings
NOISE_ENABLED = True
NOISE_PROB = 0.6
P_ERODE_DILATE = 0.5
P_OPEN_CLOSE   = 0.4
P_BLUR_THRESH  = 0.5
P_RANDOM_HOLES = 0.3
KERNEL_SIZE_RANGE = (2, 4)
BLUR_SIGMA_RANGE  = (0.8, 2.0)
HOLES_COUNT_RANGE = (1, 3)
HOLE_RADIUS_RANGE = (2, 4)

APPLY_NOISE_TO_IMAGE = False
RANDOM_SEED = 7 

if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

def rotation_matrix(theta_rad):
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    return np.array([[c, -s], [s, c]], dtype=np.float32)

def rotate_points(pts_xy, center_xy, theta_rad):
    R = rotation_matrix(theta_rad)
    return (pts_xy - center_xy) @ R.T + center_xy

def poly_to_int(pts):
    return np.round(pts).astype(np.int32)

def regular_polygon(center, radius, sides, start_angle_rad=-pi/2):
    return np.array([
        [center[0] + radius * math.cos(2*pi*i/sides + start_angle_rad),
         center[1] + radius * math.sin(2*pi*i/sides + start_angle_rad)]
        for i in range(sides)
    ], dtype=np.float32)

def star_points(center, outer_r, inner_r, points=5, start_angle_rad=-pi/2):
    pts = []
    for i in range(points*2):
        r = outer_r if i % 2 == 0 else inner_r
        theta = 2*pi*i/(points*2) + start_angle_rad
        x = center[0] + r*math.cos(theta)
        y = center[1] + r*math.sin(theta)
        pts.append([x, y])
    return np.array(pts, dtype=np.float32)

def _rand_kernel():
    k = random.randint(*KERNEL_SIZE_RANGE)
    return np.ones((k, k), np.uint8)

def _apply_erode_or_dilate(mask):
    kernel = _rand_kernel()
    return cv2.erode(mask, kernel, 1) if random.random() < 0.5 else cv2.dilate(mask, kernel, 1)

def _apply_open_or_close(mask):
    kernel = _rand_kernel()
    if random.random() < 0.5:
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

def _apply_blur_rethreshold(mask):
    sigma = random.uniform(*BLUR_SIGMA_RANGE)
    k = int(max(3, 2 * round(2 * sigma) + 1))
    blur = cv2.GaussianBlur(mask, (k, k), sigmaX=sigma, sigmaY=sigma)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def _apply_random_holes(mask):
    h, w = mask.shape
    out = mask.copy()
    for _ in range(random.randint(*HOLES_COUNT_RANGE)):
        r = random.randint(*HOLE_RADIUS_RANGE)
        ys, xs = np.where(out > 0)
        if len(xs):
            idx = random.randint(0, len(xs)-1)
            cx, cy = int(xs[idx]), int(ys[idx])
        else:
            cx, cy = random.randint(0, w-1), random.randint(0, h-1)
        cv2.circle(out, (cx, cy), r, 0, -1)
    return out

def apply_noise(mask):
    out = mask.copy()
    if random.random() < P_ERODE_DILATE: out = _apply_erode_or_dilate(out)
    if random.random() < P_OPEN_CLOSE:   out = _apply_open_or_close(out)
    if random.random() < P_BLUR_THRESH:  out = _apply_blur_rethreshold(out)
    if random.random() < P_RANDOM_HOLES: out = _apply_random_holes(out)
    return out

# Shape Creation
def draw_shape_instance(shape_name, cx, cy, base_size, scale, rot_deg, canvas_shape):
    H, W = canvas_shape
    inst_mask = np.zeros((H, W), np.uint8)
    theta = np.deg2rad(rot_deg)
    s = base_size * scale

    if shape_name == "circle":
        cv2.circle(inst_mask, (int(cx), int(cy)), int(round(s)), 255, -1)
    elif shape_name == "triangle":
        pts = np.array([[cx, cy-s],[cx-s, cy+s],[cx+s, cy+s]], np.float32)
        pts = rotate_points(pts, np.array([cx,cy]), theta)
        cv2.fillPoly(inst_mask,[poly_to_int(pts)],255)
    elif shape_name == "square":
        pts = np.array([[cx-s,cy-s],[cx+s,cy-s],[cx+s,cy+s],[cx-s,cy+s]], np.float32)
        pts = rotate_points(pts, np.array([cx,cy]), theta)
        cv2.fillPoly(inst_mask,[poly_to_int(pts)],255)
    elif shape_name == "pentagon":
        pts = regular_polygon((cx,cy), s, 5)
        pts = rotate_points(pts, np.array([cx,cy]), theta)
        cv2.fillPoly(inst_mask,[poly_to_int(pts)],255)
    elif shape_name == "hexagon":
        pts = regular_polygon((cx,cy), s, 6)
        pts = rotate_points(pts, np.array([cx,cy]), theta)
        cv2.fillPoly(inst_mask,[poly_to_int(pts)],255)
    elif shape_name == "star5":
        pts = star_points((cx,cy), s, s*0.5)
        pts = rotate_points(pts, np.array([cx,cy]), theta)
        cv2.fillPoly(inst_mask,[poly_to_int(pts)],255)
    return inst_mask

# Feature Extraction
def extract_features(binary_mask):
    cnts, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)

    area = float(cv2.contourArea(c))
    peri = float(cv2.arcLength(c, True))
    circularity = float(4*np.pi*area / (peri*peri + 1e-9))

    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = float(w) / float(h + 1e-9)

    hull = cv2.convexHull(c)
    solidity = float(area / (cv2.contourArea(hull) + 1e-9))

    vertices_hull = int(len(hull))
    extent = float(area / (w*h + 1e-9))
    eq_diameter = float(np.sqrt(4.0*area/np.pi))

    eccentricity = 0.0
    if len(c) >= 5:
        (_, _), (MA, ma), _ = cv2.fitEllipse(c)
        a, b = max(MA, ma)/2.0, min(MA, ma)/2.0
        if a > 1e-6:
            eccentricity = float(np.sqrt(max(0.0, 1.0 - (b*b)/(a*a))))

    M = cv2.moments(c)
    hu = cv2.HuMoments(M).flatten()
    hu = np.sign(hu) * np.log1p(np.abs(hu))

    return [
        area, peri, circularity, aspect_ratio, solidity,
        vertices_hull, extent, eq_diameter, eccentricity,
        *hu.tolist()
    ]

# Generation
Path(OUT_DIR, "images").mkdir(parents=True, exist_ok=True)
header = [
    "image_id","cx","cy","scale","rotation_deg",
    "area","perimeter","circularity","aspect_ratio","solidity",
    "vertices_hull","extent","eq_diameter","eccentricity",
    "hu1","hu2","hu3","hu4","hu5","hu6","hu7",
    "label"
]
rows = [header]
total_instances = 0

for i in range(NUM_IMAGES):
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
    img[:] = (30,30,30)

    shape = random.choices(SHAPES, weights=WEIGHTS, k=1)[0]

    base_size = random.randint(BASE_SIZE_MIN, BASE_SIZE_MAX)
    scale = random.uniform(SCALE_MIN, SCALE_MAX)
    rot = random.uniform(ROT_MIN_DEG, ROT_MAX_DEG)
    margin = int(math.ceil(base_size*scale*1.6))
    margin = max(8, min(margin, IMG_SIZE//2 - 1))
    cx = random.randint(margin, IMG_SIZE - margin - 1)
    cy = random.randint(margin, IMG_SIZE - margin - 1)

    inst_mask = draw_shape_instance(shape, cx, cy, base_size, scale, rot, (IMG_SIZE, IMG_SIZE))
    noisy_mask = apply_noise(inst_mask) if (NOISE_ENABLED and random.random() < NOISE_PROB) else inst_mask

    vis_mask = noisy_mask if APPLY_NOISE_TO_IMAGE else inst_mask
    img[vis_mask > 0] = (0, random.randint(100,255), random.randint(100,255))

    feats = extract_features(noisy_mask)
    if feats is None:
        continue

    rows.append([
        f"{i:04d}", cx, cy, round(scale,4), round(rot,2),
        *feats, shape
    ])
    total_instances += 1
    cv2.imwrite(str(Path(OUT_DIR, "images", f"{i:04d}_{shape}.png")), img)

    
csv_path = Path(OUT_DIR, "shapes_features.csv")
with open(csv_path, "w", newline="") as f:
    csv.writer(f).writerows(rows)

print(f"Generated {NUM_IMAGES} single-shape images ({len(SHAPES)} shape types) → {OUT_DIR}")
print(f"Total shape instances: {total_instances}")
print(f"CSV saved at: {csv_path}")
