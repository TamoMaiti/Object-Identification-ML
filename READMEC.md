```python
#CNN Model Image Analysis


import os, glob, cv2, numpy as np, tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, mixed_precision

# Set True to skip training and only run evaluation/post-processing
EVAL_ONLY = True

mixed_precision.set_global_policy("mixed_float16")

# Paths
IMG_DIR = "generated_images"
MSK_DIR = "generated_masks"
MODEL_OUT = "outputs/unet_shapes_8class_star_cross_fast.keras"
PLOT_OUT  = "outputs/loss_loglog_fast.png"
os.makedirs("outputs", exist_ok=True)

# Parameters
IMG_SIZE = 256
BATCH = 32
EPOCHS = 10
STEPS_TRAIN = 125
STEPS_VAL   = 32 
NUM_CLASSES = 8   

# Helpers
def rotate_img(img_bgr, angle_deg, border=(255,255,255)):
    h, w = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)
    return cv2.warpAffine(
        img_bgr, M, (w, h), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=border
    )

def rotate_mask(msk, angle_deg):
    h, w = msk.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)
    return cv2.warpAffine(
        msk, M, (w, h), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

# Mask Generation from Image
class PairSeq(tf.keras.utils.Sequence):
    """Loads matching (image, mask) pairs and augments with rotations/flips."""
    def __init__(self, img_dir, msk_dir, steps, batch=BATCH, size=IMG_SIZE, rot_range=(-180,180)):
        self.img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")) + glob.glob(os.path.join(img_dir, "*.jpg")))
        self.msk_files = [os.path.join(msk_dir, os.path.basename(f)) for f in self.img_files]
        if not self.img_files:
            raise RuntimeError("No images found in dataset directories.")
        self.steps, self.batch, self.size = steps, batch, size
        self.rot_lo, self.rot_hi = rot_range

    def __len__(self): return self.steps

    def __getitem__(self, idx):
        X = np.zeros((self.batch, self.size, self.size, 3), np.float32)
        Y = np.zeros((self.batch, self.size, self.size, 1), np.uint8)
        for i in range(self.batch):
            k = np.random.randint(0, len(self.img_files))
            img = cv2.imread(self.img_files[k], cv2.IMREAD_COLOR)
            msk = cv2.imread(self.msk_files[k], cv2.IMREAD_GRAYSCALE)
            # resize
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
            msk = cv2.resize(msk, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
            # rotate
            angle = float(np.random.uniform(self.rot_lo, self.rot_hi))
            img = rotate_img(img, angle)
            msk = rotate_mask(msk, angle)
            # random flips
            if np.random.rand() < 0.5:
                k90 = np.random.randint(0,4)
                if k90: img = np.rot90(img, k90).copy(); msk = np.rot90(msk, k90).copy()
            if np.random.rand() < 0.5:
                img = np.fliplr(img).copy(); msk = np.fliplr(msk).copy()
            if np.random.rand() < 0.3:
                img = np.flipud(img).copy(); msk = np.flipud(msk).copy()

            X[i] = img.astype(np.float32) / 255.0
            Y[i,:,:,0] = msk.astype(np.uint8)
        return X, Y

# Lighter U Net for Segmentation
def conv_block(x, f):
    x = layers.Conv2D(f, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(f, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    return x

def unet_multiclass(input_shape=(IMG_SIZE,IMG_SIZE,3), num_classes=NUM_CLASSES, base_filters=16):
    inputs = layers.Input(shape=input_shape)
    # Encoder
    c1 = conv_block(inputs, base_filters);     p1 = layers.MaxPool2D()(c1)
    c2 = conv_block(p1, base_filters*2);       p2 = layers.MaxPool2D()(c2)
    c3 = conv_block(p2, base_filters*4);       p3 = layers.MaxPool2D()(c3)
    c4 = conv_block(p3, base_filters*8);       p4 = layers.MaxPool2D()(c4)
    bn = conv_block(p4, base_filters*16)
    # Decoder
    u4 = layers.Conv2DTranspose(base_filters*8,2,strides=2,padding='same')(bn); u4 = layers.Concatenate()([u4,c4]); c5 = conv_block(u4,base_filters*8)
    u3 = layers.Conv2DTranspose(base_filters*4,2,strides=2,padding='same')(c5); u3 = layers.Concatenate()([u3,c3]); c6 = conv_block(u3,base_filters*4)
    u2 = layers.Conv2DTranspose(base_filters*2,2,strides=2,padding='same')(c6); u2 = layers.Concatenate()([u2,c2]); c7 = conv_block(u2,base_filters*2)
    u1 = layers.Conv2DTranspose(base_filters,  2,strides=2,padding='same')(c7); u1 = layers.Concatenate()([u1,c1]); c8 = conv_block(u1,base_filters)
    outputs = layers.Conv2D(num_classes, 1, activation='softmax', dtype='float32')(c8)
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Post Processing
def softmax_to_labels(y_pred_soft):
    return np.argmax(y_pred_soft, axis=-1).astype(np.uint8)

def relabel_polygons_by_vertices(label_mask):
    out = label_mask.copy()
    for cls_id in [2, 3, 4, 5]:
        comp = (label_mask == cls_id).astype(np.uint8)
        if comp.max() == 0:
            continue
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) < 10:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            v = len(approx)
            target = {3:3, 4:2, 5:4, 6:5}.get(v, cls_id)
            cv2.drawContours(out, [c], -1, int(target), thickness=-1)
    return out

def fill_polygons_confidently(y_soft, y_pred,
                              thr_fg=0.45, min_area=500,
                              conf_keep=0.55, cover_keep=0.50,
                              conf_geom_min=0.30):
H, W, C = y_soft.shape
    out = y_pred.copy()

    p_bg = y_soft[..., 0]
    fg = ((1.0 - p_bg) > thr_fg).astype(np.uint8)
    if fg.max() == 0:
        return out
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    contours, hierarchy = cv2.findContours(fg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return out
    hierarchy = hierarchy[0] 

    def region_mask_for_outer(i):
        m = np.zeros((H, W), np.uint8)
        cv2.drawContours(m, [contours[i]], -1, 1, thickness=-1)
        child = hierarchy[i][2]
        while child != -1:
            cv2.drawContours(m, [contours[child]], -1, 0, thickness=-1)
            child = hierarchy[child][0]
        return m

    for i, h in enumerate(hierarchy):
        if h[3] != -1:
            continue

        c = contours[i]
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        region = region_mask_for_outer(i)
        idx = (region > 0)

        vals = out[idx]
        vals_nbg = vals[vals != 0]
        mode_cls = int(np.bincount(vals_nbg).argmax()) if vals_nbg.size else 0
        cover = (vals == mode_cls).mean() if mode_cls != 0 else 0.0
        conf = float(y_soft[..., mode_cls][idx].mean()) if mode_cls != 0 else 0.0

        peri  = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * max(peri, 1.0), True)
        v = len(approx)
        hull = cv2.convexHull(c, returnPoints=True)
        hull_area = max(cv2.contourArea(hull), 1.0)
        solidity = area / hull_area
        circ = 4.0 * np.pi * area / (peri * peri + 1e-6)

        looks_convex_polygon = (solidity > 0.90 and (3 <= v <= 6 or circ > 0.80))
        cross_but_convex = (mode_cls == 7 and looks_convex_polygon)

        keep_model = (mode_cls != 0) and ((conf >= conf_keep and cover >= cover_keep))
        if keep_model and not cross_but_convex:
            out[idx] = mode_cls
            continue

        geom_cls = 0
        if circ > 0.88:
            geom_cls = 1  # circle
        elif v in (3, 4, 5, 6):
            geom_cls = {3:3, 4:2, 5:4, 6:5}[v]

        if geom_cls != 0 and (looks_convex_polygon or cross_but_convex):
            geom_conf = float(y_soft[..., geom_cls][idx].mean())
            if geom_conf >= conf_geom_min or cross_but_convex:
                out[idx] = geom_cls
                continue

        if mode_cls != 0:
            out[idx] = mode_cls

    return out

def mean_iou_macro(y_true, y_pred, n=NUM_CLASSES):
    from sklearn.metrics import jaccard_score
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    return jaccard_score(y_true, y_pred, average='macro', labels=list(range(n)))

#Main
def main():
    if EVAL_ONLY and os.path.exists(MODEL_OUT):
        print(f"Loading saved model from {MODEL_OUT} (eval-only)...")
        model = tf.keras.models.load_model(MODEL_OUT)
    else:
        print("Building a fresh model...")
        model = unet_multiclass(base_filters=16)

    train_ds = PairSeq(IMG_DIR, MSK_DIR, steps=STEPS_TRAIN, batch=BATCH, size=IMG_SIZE)
    val_ds   = PairSeq(IMG_DIR, MSK_DIR, steps=STEPS_VAL,   batch=BATCH, size=IMG_SIZE)

    if not EVAL_ONLY:
        cbs = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(MODEL_OUT, save_best_only=True)
        ]
        print("Starting training...")
        history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cbs)
        model.save(MODEL_OUT)
        print(f"Saved model to {MODEL_OUT}")

        # Plot Loss (Log-Log)
        plt.figure(figsize=(6,5))
        plt.loglog(history.history["loss"], label="train_loss")
        plt.loglog(history.history["val_loss"], label="val_loss")
        plt.xlabel("Epoch (log scale)")
        plt.ylabel("Loss (log scale)")
        plt.title("Training vs Validation Loss (log–log) — Light U-Net")
        plt.legend()
        plt.grid(True, which="both", ls="--", lw=0.5)
        plt.tight_layout()
        plt.savefig(PLOT_OUT, dpi=150)
        plt.close()
        print(f"Saved log–log loss plot to {PLOT_OUT}")
    else:
        print("Skipping training; running evaluation only.")

    # Post Processing and Evaluation
    X_val, Y_val = val_ds[0]                 
    y_soft = model.predict(X_val, verbose=0) 
    y_pred = softmax_to_labels(y_soft)      
    y_true = np.squeeze(Y_val, axis=-1)     

    y_filled = np.stack([
        fill_polygons_confidently(y_soft[i], y_pred[i],
                                  thr_fg=0.45, min_area=500,
                                  conf_keep=0.55, cover_keep=0.50,
                                  conf_geom_min=0.30)
        for i in range(y_pred.shape[0])
    ], axis=0)

    y_post = np.stack([relabel_polygons_by_vertices(m) for m in y_filled], axis=0)

    miou_raw  = mean_iou_macro(y_true, y_pred,  NUM_CLASSES)
    miou_fill = mean_iou_macro(y_true, y_filled, NUM_CLASSES)
    miou_post = mean_iou_macro(y_true, y_post,  NUM_CLASSES)
    print(f"mIoU (raw):    {miou_raw:.4f}")
    print(f"mIoU (filled): {miou_fill:.4f}")
    print(f"mIoU (post):   {miou_post:.4f}")

    os.makedirs("outputs/preview_fast", exist_ok=True)
    def colorize(mask):
        scaled = (mask.astype(np.float32) * (255.0 / (NUM_CLASSES - 1))).astype(np.uint8)
        return cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)

    for i in range(min(4, X_val.shape[0])):
        vis = np.hstack([
            (X_val[i] * 255).astype(np.uint8),
            colorize(y_pred[i]),
        ])
        cv2.putText(vis, "Input | Pred", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, "Input | Pred", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(f"outputs/preview_fast/sample_{i}_panel.png", vis)
    
    print("Saved previews to outputs/preview_fast/")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only", action="store_true", help="Skip training; load MODEL_OUT and run evaluation")
    args = ap.parse_args()
    if args.eval_only:
        EVAL_ONLY = True
    main()
