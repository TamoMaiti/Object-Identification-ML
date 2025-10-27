```python
#Dataset Generation for CNN Model

import os, cv2, numpy as np, random, math, multiprocessing

# Output
OUTPUT_IMG_DIR = "generated_images"
OUTPUT_MSK_DIR = "generated_masks"
N_IMAGES = 1000
OUT_W, OUT_H = 512, 512

# Classes 
CID_BG=0; CID_CIRCLE=1; CID_SQUARE=2; CID_TRIANGLE=3
CID_PENTAGON=4; CID_HEXAGON=5; CID_STAR=6; CID_XCROSS=7
SHAPES=[
    ("circle",CID_CIRCLE),
    ("square",CID_SQUARE),
    ("triangle",CID_TRIANGLE),
    ("pentagon",CID_PENTAGON),
    ("hexagon",CID_HEXAGON),
    ("star",CID_STAR),
    ("xcross",CID_XCROSS)
]

BASE_SIZE_RANGE=(22,70)
ANGLE_RANGE_DEG=(-45,45)
SCALE_RANGE=(0.8,1.25)

# Density Settings
DENSITY_BUCKETS=[
    {"prob":0.3,"fill":(0.10,0.20),"shapes":(1,6)},
    {"prob":0.5,"fill":(0.25,0.45),"shapes":(5,12)},
    {"prob":0.2,"fill":(0.50,0.60),"shapes":(10,20)},
]

ALLOW_INSCRIPTIONS=True
INSCRIPTION_PROB=0.35
INSCRIPTION_SCALE_RANGE=(0.45,0.75)

REDUCE_OVERLAP=True
MIN_SEP_FACTOR=0.35
MAX_IoU=0.05
PLACEMENT_TRIES=80
FILL_TRIES_LIMIT=500

WHITE=(255,255,255)
RED=(0,0,255)
BLUE=(255,0,0)
rng=np.random.default_rng(42)

# Helpers
def rand_angle_rad(lo,hi): return math.radians(float(rng.uniform(lo,hi)))
def rand_scale(lo,hi):    return float(rng.uniform(lo,hi))

def poly_points(center,radius,sides,theta0):
    cx,cy=center
    return np.array([
        (int(cx+radius*math.cos(theta0+2*math.pi*i/sides)),
         int(cy+radius*math.sin(theta0+2*math.pi*i/sides)))
        for i in range(sides)
    ],np.int32)

def star_points(center, outer_r, angle_rad):
    cx, cy = center
    inner_r = outer_r * 0.5
    pts = []
    base_deg = -90.0 + math.degrees(angle_rad)
    for i in range(10):
        r = outer_r if i % 2 == 0 else inner_r
        theta = math.radians(base_deg + i * 36.0)
        x = int(round(cx + r * math.cos(theta)))
        y = int(round(cy + r * math.sin(theta)))
        pts.append((x,y))
    return np.array(pts, np.int32)

def draw_rotated_rect(mask, center, w, h, angle_deg, color_val):
    cx, cy = center
    rect = ((cx, cy), (float(w), float(h)), float(angle_deg))
    box = cv2.boxPoints(rect)  # 4x2
    box = np.int32(np.round(box))
    cv2.fillConvexPoly(mask, box, color_val)

def draw_xcross(img, msk, center, size, color, cls_id, angle_rad):
    H, W = msk.shape[:2]
    temp = np.zeros((H, W), np.uint8)
    bar_len = int(size * 2.2)
    bar_w   = max(2, int(size * 0.30))
    base_deg = math.degrees(angle_rad)
    draw_rotated_rect(temp, center, bar_len, bar_w, base_deg + 45.0, 255)
    draw_rotated_rect(temp, center, bar_len, bar_w, base_deg - 45.0, 255)
    img[temp > 0] = color
    msk[temp > 0] = int(cls_id)

def draw_star(img, msk, center, size, color, cls_id, angle_rad):
    pts = star_points(center, size, angle_rad)
    cv2.fillPoly(img, [pts], color)    
    cv2.fillPoly(msk, [pts], int(cls_id))

def draw_shape(img, msk, shape_type, center, base_size, color, cls_id, angle_rad=None, scale=None):
    if angle_rad is None: angle_rad=rand_angle_rad(*ANGLE_RANGE_DEG)
    if scale is None:     scale=rand_scale(*SCALE_RANGE)
    size=max(3,int(round(base_size*scale)))

    if shape_type=="circle":
        cv2.circle(img,center,size,color,-1)
        cv2.circle(msk,center,size,int(cls_id),-1)
        return size,angle_rad
    elif shape_type=="star":
        draw_star(img,msk,center,size,color,cls_id,angle_rad); return size,angle_rad
    elif shape_type=="xcross":
        draw_xcross(img,msk,center,size,color,cls_id,angle_rad); return size,angle_rad
    elif shape_type=="square":
        pts=poly_points(center,size,4,angle_rad)
    elif shape_type=="triangle":
        pts=poly_points(center,size,3,angle_rad)
    elif shape_type=="pentagon":
        pts=poly_points(center,size,5,angle_rad)
    elif shape_type=="hexagon":
        pts=poly_points(center,size,6,angle_rad)
    else:
        raise ValueError(shape_type)

    cv2.fillConvexPoly(img,pts,color)
    cv2.fillConvexPoly(msk,pts,int(cls_id))
    return size,angle_rad

def rasterize(shape_type, center, size, canvas_shape, angle_rad):
    H,W=canvas_shape[:2]
    m=np.zeros((H,W),np.uint8)
    if shape_type=="circle":
        cv2.circle(m,center,size,255,-1); return m
    sides_map={"square":4,"triangle":3,"pentagon":5,"hexagon":6}
    if shape_type in sides_map:
        pts=poly_points(center,size,sides_map[shape_type],angle_rad)
        cv2.fillConvexPoly(m,pts,255)
    elif shape_type=="star":
        pts = star_points(center, size, angle_rad)
        cv2.fillPoly(m, [pts], 255)
    elif shape_type=="xcross":
        base_deg = math.degrees(angle_rad)
        draw_rotated_rect(m, center, int(size*2.2), max(2,int(size*0.30)), base_deg + 45.0, 255)
        draw_rotated_rect(m, center, int(size*2.2), max(2,int(size*0.30)), base_deg - 45.0, 255)
    return m

def iou(a,b):
    inter=np.logical_and(a>0,b>0).sum()
    uni=np.logical_or(a>0,b>0).sum()
    return (inter/uni) if uni else 0.0

def pick_density_bucket():
    ps=[b["prob"] for b in DENSITY_BUCKETS]
    idx=min(int(np.searchsorted(np.cumsum(ps),rng.random())),len(DENSITY_BUCKETS)-1)
    return DENSITY_BUCKETS[idx]

# Image Generation
def generate_image_and_mask(W=OUT_W,H=OUT_H):
    img=np.full((H,W,3),WHITE,np.uint8)
    msk=np.zeros((H,W),np.uint8)
    bucket=pick_density_bucket()
    target_fill=float(rng.uniform(*bucket["fill"]))
    n_shapes=int(rng.integers(bucket["shapes"][0], bucket["shapes"][1]+1))

    outer_masks, outer_meta=[],[]
    placed, tries=0,0
    while placed<n_shapes and tries<n_shapes*PLACEMENT_TRIES:
        tries+=1
        shape_type,cls_id=random.choice(SHAPES)
        color=RED if rng.random()<0.5 else BLUE
        base_size=int(rng.integers(BASE_SIZE_RANGE[0], BASE_SIZE_RANGE[1]+1))
        cx=int(rng.integers(base_size+10,W-base_size-10))
        cy=int(rng.integers(base_size+10,H-base_size-10))
        theta=rand_angle_rad(*ANGLE_RANGE_DEG)
        sc=rand_scale(*SCALE_RANGE)
        size_used=max(3,int(round(base_size*sc)))
        ok=True; cand=None

        if REDUCE_OVERLAP and outer_meta:
            for (ox,oy,os) in outer_meta:
                if (cx-ox)**2+(cy-oy)**2 < (MIN_SEP_FACTOR*(size_used+os))**2:
                    ok=False; break
            if ok and MAX_IoU is not None:
                cand=rasterize(shape_type,(cx,cy),size_used,img.shape,theta)
                for om in outer_masks:
                    if iou(cand,om)>MAX_IoU:
                        ok=False; break
        if not ok: continue

        size_used,_=draw_shape(img,msk,shape_type,(cx,cy),base_size,color,cls_id,angle_rad=theta,scale=sc)
        if cand is None: cand=rasterize(shape_type,(cx,cy),size_used,img.shape,theta)
        outer_masks.append(cand); outer_meta.append((cx,cy,size_used))
        placed+=1

        if ALLOW_INSCRIPTIONS and rng.random()<INSCRIPTION_PROB:
            inner_type,inner_cid=random.choice(SHAPES)
            inner_color=BLUE if np.array_equal(color,RED) else RED
            inner_base=max(3,int(round(size_used*rng.uniform(*INSCRIPTION_SCALE_RANGE))))
            inner_theta=rand_angle_rad(*ANGLE_RANGE_DEG)
            draw_shape(img,msk,inner_type,(cx,cy),inner_base,inner_color,inner_cid,angle_rad=inner_theta,scale=1.0)

    extra=0
    while extra<FILL_TRIES_LIMIT:
        if (msk>0).mean()>=target_fill: break
        extra+=1
        shape_type,cls_id=random.choice(SHAPES)
        color=RED if rng.random()<0.5 else BLUE
        base_size=int(rng.integers(max(16,BASE_SIZE_RANGE[0]-4), BASE_SIZE_RANGE[1]+1))
        cx=int(rng.integers(base_size+8,W-base_size-8))
        cy=int(rng.integers(base_size+8,H-base_size-8))
        theta=rand_angle_rad(*ANGLE_RANGE_DEG)
        sc=rand_scale(*SCALE_RANGE)
        size_used=max(3,int(round(base_size*sc)))
        ok=True; cand=None
        if REDUCE_OVERLAP and outer_meta:
            for (ox,oy,os) in outer_meta:
                if (cx-ox)**2+(cy-oy)**2 < (MIN_SEP_FACTOR*(size_used+os))**2:
                    ok=False; break
            if ok and MAX_IoU is not None:
                cand=rasterize(shape_type,(cx,cy),size_used,img.shape,theta)
                for om in outer_masks:
                    if iou(cand,om)>MAX_IoU:
                        ok=False; break
        if not ok: continue
        size_used,_=draw_shape(img,msk,shape_type,(cx,cy),base_size,color,cls_id,angle_rad=theta,scale=sc)
        if cand is None: cand=rasterize(shape_type,(cx,cy),size_used,img.shape,theta)
        outer_masks.append(cand); outer_meta.append((cx,cy,size_used))

    return img,msk

def worker(i):
    img,msk=generate_image_and_mask(OUT_W,OUT_H)
    fname=f"shapes_{i:06d}.png"
    cv2.imwrite(os.path.join(OUTPUT_IMG_DIR,fname),img)
    cv2.imwrite(os.path.join(OUTPUT_MSK_DIR,fname),msk)

def main():
    os.makedirs(OUTPUT_IMG_DIR,exist_ok=True)
    os.makedirs(OUTPUT_MSK_DIR,exist_ok=True)
    print(f"Generating {N_IMAGES} images (star + X-cross enabled)...")
    with multiprocessing.Pool(processes=max(1,multiprocessing.cpu_count()-1)) as pool:
        for _ in pool.imap_unordered(worker, range(1, N_IMAGES+1)):
            pass
    print("Done.")

if __name__=="__main__":
    main()
