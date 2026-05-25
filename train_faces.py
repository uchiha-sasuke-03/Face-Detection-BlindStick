"""
Face Embedding Training Script (v6 - Haar Cascade + face_recognition)
=====================================================================
Uses OpenCV Haar Cascade for face detection and dlib (via face_recognition)
for 128-dimensional face embeddings. Lightweight enough for Raspberry Pi 4.

Strategy:
  - Haar Cascade: fast face detection, ships with OpenCV
  - face_recognition (dlib): 128-dim embeddings, excellent accuracy
  - 3-pass detection: original → CLAHE enhanced → histogram equalized
  - Data augmentation: 13 variants per image for maximum robustness
  - num_jitters=3 during encoding for more stable embeddings
  - Stores per-person embeddings + averaged centroid for fast matching

Usage:
    python train_faces.py
"""

import os
import sys
import pickle
import cv2
import numpy as np
import time
import face_recognition

# ── Config ──────────────────────────────────────────────────────────────────
KNOWN_FACES_DIR  = "known_faces"
DB_OUTPUT_PATH   = "known_faces/face_embeddings_v6.pkl"
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# Folders to skip (generic names, not a person)
SKIP_FOLDERS = {"Blind Stick Face Recog", "__pycache__"}

# Haar Cascade path (ships with OpenCV)
HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Number of jitters for face_recognition encoding (higher = more accurate, slower)
NUM_JITTERS = 3


def init_detector():
    """Initialize the Haar Cascade face detector."""
    print(f"  Loading Haar Cascade face detector...")
    cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    if cascade.empty():
        print("ERROR: Failed to load Haar Cascade classifier.")
        sys.exit(1)
    print(f"  Haar Cascade loaded from: {HAAR_CASCADE_PATH}")
    return cascade


def collect_images(root_dir):
    """Collect (image_path, person_name) pairs from known_faces/."""
    entries = []
    for item in sorted(os.listdir(root_dir)):
        item_path = os.path.join(root_dir, item)

        if os.path.isfile(item_path) and item.lower().endswith(IMAGE_EXTENSIONS):
            name = os.path.splitext(item)[0]
            entries.append((item_path, name))

        elif os.path.isdir(item_path):
            if item in SKIP_FOLDERS or item.startswith('.'):
                print(f"  [SKIP] Folder '{item}'")
                continue
            folder_name = item
            for fname in sorted(os.listdir(item_path)):
                fpath = os.path.join(item_path, fname)
                if os.path.isfile(fpath) and fname.lower().endswith(IMAGE_EXTENSIONS):
                    entries.append((fpath, folder_name))

    return entries


def augment_image(img_bgr):
    """
    Generate augmented versions of an image for lighting and pose robustness.
    Returns list of BGR images (including original).
    13 variants total: original + flip + 4 gamma + 2 brightness + CLAHE
                       + 2 rotations + blur + noise
    """
    augmented = [img_bgr]

    # 1. Horizontal flip
    augmented.append(cv2.flip(img_bgr, 1))

    # 2. Gamma corrections (simulate different lighting)
    for gamma in [0.4, 0.6, 1.5, 2.0]:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in range(256)]).astype("uint8")
        augmented.append(cv2.LUT(img_bgr, table))

    # 3. Brightness adjustments
    for alpha, beta in [(0.6, -20), (1.4, 20)]:
        augmented.append(cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta))

    # 4. CLAHE on the luminance channel
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    augmented.append(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))

    # 5. Slight rotations (±10°) for pose robustness
    h, w = img_bgr.shape[:2]
    center = (w // 2, h // 2)
    for angle in [-10, 10]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img_bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        augmented.append(rotated)

    # 6. Gaussian blur (simulates slight motion blur / out-of-focus)
    augmented.append(cv2.GaussianBlur(img_bgr, (5, 5), 0))

    # 7. Gaussian noise (simulates sensor noise in low light)
    noise = np.random.normal(0, 12, img_bgr.shape).astype(np.int16)
    noisy = np.clip(img_bgr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    augmented.append(noisy)

    return augmented


def detect_face_haar(img_bgr, cascade):
    """
    Detect faces using Haar Cascade with tuned parameters.
    Returns list of (x, y, w, h) tuples.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces if len(faces) > 0 else []


def enhance_clahe(img_bgr):
    """Apply CLAHE enhancement for low-light images."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def enhance_histeq(img_bgr):
    """Apply histogram equalization for contrast improvement."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)


def extract_embedding(img_bgr, cascade):
    """
    Detect the largest face in a BGR image and return its 128-dim embedding.
    Uses 3-pass detection: original → CLAHE → histogram equalization.
    Returns (embedding_np, info_str) or (None, error_str).
    """
    # Pass 1: Original image
    faces = detect_face_haar(img_bgr, cascade)
    source = img_bgr

    # Pass 2: CLAHE enhanced
    if len(faces) == 0:
        enhanced = enhance_clahe(img_bgr)
        faces = detect_face_haar(enhanced, cascade)
        source = enhanced

    # Pass 3: Histogram equalization
    if len(faces) == 0:
        histeq = enhance_histeq(img_bgr)
        faces = detect_face_haar(histeq, cascade)
        source = histeq

    if len(faces) == 0:
        return None, "no face detected (3-pass)"

    # Pick the largest face (by area)
    largest = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = largest

    # Convert Haar (x, y, w, h) to face_recognition format (top, right, bottom, left)
    top = y
    right = x + w
    bottom = y + h
    left = x
    face_location = (top, right, bottom, left)

    # Convert BGR to RGB for face_recognition
    img_rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

    # Get 128-dim face encoding using dlib with jittering
    encodings = face_recognition.face_encodings(
        img_rgb,
        known_face_locations=[face_location],
        num_jitters=NUM_JITTERS,
        model="large"
    )

    if len(encodings) == 0:
        return None, "face detected but encoding failed"

    emb_np = encodings[0].astype(np.float32)

    # L2 normalize
    norm = np.linalg.norm(emb_np)
    if norm > 0:
        emb_np = emb_np / norm

    return emb_np, f"face=({w}x{h})"


def train():
    print("=" * 62)
    print("  SMART BLIND STICK - FACE TRAINING v6 (Haar + face_recognition)")
    print("=" * 62)
    print(f"  Embedding     : dlib face_recognition (128-dim)")
    print(f"  Detector      : OpenCV Haar Cascade")
    print(f"  num_jitters   : {NUM_JITTERS}")
    print(f"  Augmentation  : flip + gamma + brightness + CLAHE + rotate + blur + noise (13x)")
    print(f"  Detection     : 3-pass (original -> CLAHE -> histogram eq.)")
    print(f"  Source dir    : {os.path.abspath(KNOWN_FACES_DIR)}")
    print("=" * 62)

    if not os.path.exists(KNOWN_FACES_DIR):
        print(f"ERROR: '{KNOWN_FACES_DIR}' not found.")
        sys.exit(1)

    # Initialize detector
    cascade = init_detector()

    print("\nScanning images...")
    entries = collect_images(KNOWN_FACES_DIR)
    if not entries:
        print("ERROR: No images found.")
        sys.exit(1)

    # Group by person
    people = {}
    for path, name in entries:
        people.setdefault(name, []).append(path)

    print(f"\nFound {len(entries)} images for {len(people)} people:")
    for name, paths in people.items():
        print(f"  >> {name}: {len(paths)} image(s)")

    print(f"\n--- Extracting face_recognition embeddings ---\n")
    database = {}
    success_count = 0
    fail_count = 0
    start_time = time.time()

    for i, (image_path, name) in enumerate(entries, 1):
        short = os.path.basename(image_path)
        if len(short) > 40:
            short = short[:37] + "..."
        print(f"  [{i:2d}/{len(entries)}] {name}/{short}")

        img = cv2.imread(image_path)
        if img is None:
            fail_count += 1
            print(f"           SKIP - cannot read")
            continue

        # Generate augmented versions
        augmented_images = augment_image(img)
        img_success = 0

        for j, aug_img in enumerate(augmented_images):
            emb, info = extract_embedding(aug_img, cascade)
            if emb is not None:
                database.setdefault(name, []).append(emb)
                img_success += 1

        if img_success > 0:
            success_count += 1
            print(f"           OK - {img_success}/{len(augmented_images)} variants embedded")
        else:
            fail_count += 1
            print(f"           SKIP - no face in any variant")

    elapsed = time.time() - start_time

    if not database:
        print("\nERROR: No faces extracted. Check your images.")
        sys.exit(1)

    # Compute per-person average centroid
    print(f"\nComputing per-person averaged embeddings...")
    final_db = {}
    for name, emb_list in database.items():
        avg = np.mean(emb_list, axis=0).astype(np.float32)
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg = avg / norm
        final_db[name] = {
            'embeddings': emb_list,
            'average': avg,
            'count': len(emb_list),
        }

    # Save
    output = {
        'model': 'dlib-face_recognition-128d',
        'embedding_dim': 128,
        'people': final_db,
        'trained_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(DB_OUTPUT_PATH, 'wb') as f:
        pickle.dump(output, f)

    size_kb = os.path.getsize(DB_OUTPUT_PATH) / 1024
    total_embs = sum(d['count'] for d in final_db.values())

    print(f"\n{'=' * 62}")
    print(f"  TRAINING COMPLETE")
    print(f"{'=' * 62}")
    print(f"  People registered : {len(final_db)}")
    print(f"  Total embeddings  : {total_embs}")
    print(f"  Source images OK  : {success_count}")
    print(f"  Skipped images    : {fail_count}")
    print(f"  Time              : {elapsed:.1f}s")
    print(f"  DB size           : {size_kb:.1f} KB")
    print(f"  Saved to          : {DB_OUTPUT_PATH}")
    print(f"{'=' * 62}")
    for name, data in final_db.items():
        print(f"  [OK] {name}: {data['count']} embedding(s)")
    print(f"\nRestart main.py to use the new face database.")


if __name__ == '__main__':
    train()
