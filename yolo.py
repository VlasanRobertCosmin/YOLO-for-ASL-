import cv2, mediapipe as mp
from pathlib import Path

# INPUT (your raw images by class folders)
IMG_DIR = Path("data/asl_images")      

# OUTPUT (flat YOLO dirs are fine)
OUT_IMG = Path("data/yolo/images")
OUT_LBL = Path("data/yolo/labels")
OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_LBL.mkdir(parents=True, exist_ok=True)

# Classes: A–Z + special tokens
CLASSES = [*list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), "del", "nothing", "space"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

# MediaPipe Hands (static images)
mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1)

def keypoints_bbox(norm_xy):
    xs = [x for x, _ in norm_xy]; ys = [y for _, y in norm_xy]
    xmin, xmax = max(0.0, min(xs)), min(1.0, max(xs))
    ymin, ymax = max(0.0, min(ys)), min(1.0, max(ys))
    w = max(1e-6, xmax - xmin); h = max(1e-6, ymax - ymin)
    xc, yc = xmin + w / 2, ymin + h / 2
    return xc, yc, w, h

def class_from_path(p: Path):
    cls = p.parent.name.lower()
    # normalize any odd names here if needed (e.g., "spacebar" -> "space")
    if cls == "spacebar": cls = "space"
    # map back to canonical casing
    for c in CLASSES:
        if c.lower() == cls:
            return c
    return None

def process_image(p: Path):
    img = cv2.imread(str(p))
    if img is None:
        print(f"[WARN] cannot read: {p}")
        return

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = mp_hands.process(rgb)
    if not res.multi_hand_landmarks:
        # skip images without a detected hand
        # (or you can write a zero-area box if you prefer)
        return

    lm = res.multi_hand_landmarks[0].landmark
    norm_xy = [(l.x, l.y) for l in lm]  # already in [0,1] relative to original image size
    xc, yc, ww, hh = keypoints_bbox(norm_xy)

    cls = class_from_path(p)
    if cls is None:
        print(f"[SKIP] unknown class folder: {p.parent.name}")
        return
    cid = CLASS_TO_ID[cls]

    # avoid filename collisions by prefixing class
    out_img = OUT_IMG / f"{cls}_{p.name}"
    out_lbl = OUT_LBL / f"{cls}_{p.stem}.txt"

    cv2.imwrite(str(out_img), img)
    with open(out_lbl, "w") as f:
        f.write(f"{cid} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n")

if __name__ == "__main__":
    imgs = list(IMG_DIR.rglob("*.jpg")) + list(IMG_DIR.rglob("*.jpeg")) + list(IMG_DIR.rglob("*.png"))
    for p in imgs:
        process_image(p)
    mp_hands.close()
    print("Done.")
