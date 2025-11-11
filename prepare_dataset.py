# prepare_asl_dataset.py
# ------------------------------------------------------------
# One-shot dataset prep for ASL Alphabet (Kaggle-style) -> YOLO.
# - Moves folders into DIZERTATIE/dataset/{train,test}
# - Uses MediaPipe Hands to create YOLO bboxes from 21 keypoints
# - Splits into train/val and writes asl.yaml
# ------------------------------------------------------------
import os
import shutil
from pathlib import Path
import random
import cv2
import numpy as np

# Lazy import mediapipe with helpful error if missing
try:
    import mediapipe as mp
except Exception as e:
    raise SystemExit(
        "mediapipe is required. Install with:\n  pip install mediapipe opencv-python ultralytics\n\nOriginal error: "
        + str(e)
    )

# ------------- CONFIG (change if you want) -------------------
PROJECT_ROOT = Path(__file__).resolve().parent  # run from folder containing archive/ and DIZERTATIE/
ARCHIVE_DIR  = PROJECT_ROOT / "archive"
TARGET_DS    = PROJECT_ROOT / "dizertatie" / "dataset"       # final dataset base
YOLO_ROOT    = TARGET_DS / "yolo"                            # yolo export root
VAL_RATIO    = 0.1                                           # 10% validation split
RANDOM_SEED  = 42
IMG_EXTS     = {".jpg", ".jpeg", ".png"}
CLASSES      = [*list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), "del", "nothing", "space"]
CLASS_TO_ID  = {c:i for i,c in enumerate(CLASSES)}
# -------------------------------------------------------------

def ensure_dirs():
    (PROJECT_ROOT / "DIZERTATIE").mkdir(exist_ok=True)
    TARGET_DS.mkdir(parents=True, exist_ok=True)
    (YOLO_ROOT / "images" / "train").mkdir(parents=True, exist_ok=True)
    (YOLO_ROOT / "images" / "val").mkdir(parents=True, exist_ok=True)
    (YOLO_ROOT / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (YOLO_ROOT / "labels" / "val").mkdir(parents=True, exist_ok=True)

def find_source_folders():
    """
    Try to locate Kaggle-style folders inside archive/.
    Expected:
      archive/asl_alphabet_train/asl_alphabet_train/<A,B,...,space,del,nothing>/
      archive/asl_alphabet_test/asl_alphabet_test/*.jpg
    Returns (train_src, test_src) paths.
    """
    train_candidates = [
        ARCHIVE_DIR / "asl_alphabet_train" / "asl_alphabet_train",
        ARCHIVE_DIR / "asl_alphabet_train",
    ]
    test_candidates = [
        ARCHIVE_DIR / "asl_alphabet_test" / "asl_alphabet_test",
        ARCHIVE_DIR / "asl_alphabet_test",
    ]

    train_src = next((p for p in train_candidates if p.exists()), None)
    test_src  = next((p for p in test_candidates  if p.exists()), None)
    return train_src, test_src

def move_into_target(train_src: Path, test_src: Path):
    """Move (or copy) raw folders into DIZERTATIE/dataset/{train,test} if not present."""
    target_train = TARGET_DS / "train"
    target_test  = TARGET_DS / "test"

    if not target_train.exists():
        print(f"→ Moving train: {train_src} → {target_train}")
        shutil.move(str(train_src), str(target_train))
    else:
        print(f"✓ Train already in place: {target_train}")

    if not target_test.exists() and test_src and test_src.exists():
        print(f"→ Moving test: {test_src} → {target_test}")
        shutil.move(str(test_src), str(target_test))
    else:
        print(f"✓ Test already in place (or not found): {target_test}")

def iter_images(folder: Path):
    for p in folder.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            yield p

def infer_class_from_path(p: Path):
    """
    For train images: parent folder name is the class (A,B,...,space,del,nothing)
    """
    cls = p.parent.name.lower()
    # normalize names
    if cls == "spacebar": cls = "space"
    if cls not in [c.lower() for c in CLASSES]:
        return None
    # map back to canonical case
    for c in CLASSES:
        if c.lower() == cls:
            return c
    return None

def mediapipe_bbox(img_bgr):
    """
    Returns normalized YOLO bbox (xc,yc,w,h) from MediaPipe 21 landmarks.
    None if no hand found.
    """
    mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = mp_hands.process(rgb)
    mp_hands.close()
    if not res.multi_hand_landmarks:
        return None
    lm = res.multi_hand_landmarks[0].landmark
    xs = [l.x for l in lm]; ys = [l.y for l in lm]
    xmin, xmax = max(0.0, min(xs)), min(1.0, max(xs))
    ymin, ymax = max(0.0, min(ys)), min(1.0, max(ys))
    w = max(1e-6, xmax - xmin); h = max(1e-6, ymax - ymin)
    xc, yc = xmin + w/2, ymin + h/2
    return xc, yc, w, h

def write_yolo_pair(img_path: Path, split: str, class_id: int):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] Could not read image: {img_path}")
        return False

    bbox = mediapipe_bbox(img)
    if bbox is None:
        print(f"[SKIP] No hand found: {img_path}")
        return False

    xc, yc, w, h = bbox

    # destination paths
    out_img = YOLO_ROOT / "images" / split / img_path.name
    out_lbl = YOLO_ROOT / "labels" / split / (img_path.stem + ".txt")

    # copy image
    cv2.imwrite(str(out_img), img)
    # write label
    out_lbl.write_text(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
    return True

def build_splits():
    """
    Make a list of (image_path, class_id) from TARGET_DS/train,
    shuffle and split into train/val.
    """
    pairs = []
    for p in iter_images(TARGET_DS / "train"):
        cls = infer_class_from_path(p)
        if not cls: 
            continue
        cid = CLASS_TO_ID[cls]
        pairs.append((p, cid))

    random.Random(RANDOM_SEED).shuffle(pairs)
    n = len(pairs)
    n_val = int(n * VAL_RATIO)
    val = pairs[:n_val]
    train = pairs[n_val:]
    return train, val

def export_yaml():
    yaml_path = PROJECT_ROOT / "asl.yaml"
    names_block = "\n".join([f"  {i}: {c}" for c,i in CLASS_TO_ID.items()])
    yaml_text = f"""path: {YOLO_ROOT.as_posix()}
train: images/train
val: images/val
names:
{names_block}
"""
    yaml_path.write_text(yaml_text)
    print(f"✓ Wrote {yaml_path}")

def main():
    print("=== ASL dataset one-shot prep ===")
    ensure_dirs()

    train_src, test_src = find_source_folders()
    if not train_src or not train_src.exists():
        raise SystemExit("Could not find the train folder under archive/. "
                         "Make sure the Kaggle archive is unzipped inside ./archive")
    move_into_target(train_src, test_src)

    # Build split lists
    train_pairs, val_pairs = build_splits()
    print(f"Found {len(train_pairs)+len(val_pairs)} labeled images.")
    print(f"→ Train: {len(train_pairs)} | Val: {len(val_pairs)}")

    # Convert each image to YOLO (images + labels)
    ok_train = ok_val = 0
    for p, cid in train_pairs:
        ok_train += write_yolo_pair(p, "train", cid)
    for p, cid in val_pairs:
        ok_val += write_yolo_pair(p, "val", cid)
    print(f"✓ Wrote YOLO train: {ok_train} | val: {ok_val}")

    export_yaml()
    print("\nAll set!")
    print("Train YOLO with for example:")
    print(f"  yolo detect train data={ (PROJECT_ROOT / 'asl.yaml').as_posix() } model=yolov8n.pt imgsz=640 epochs=50 batch=16")

if __name__ == "__main__":
    main()
