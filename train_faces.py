"""
Face Embedding Training Script (v6 - DeepFace Facenet512)
==========================================================
Uses DeepFace with Facenet512 model for 512-dimensional face embeddings
and OpenCV's Haar Cascade for fast, lightweight face detection.
Optimized for Raspberry Pi 4 deployment — no PyTorch required.

Strategy:
  - OpenCV Haar Cascade: fast, lightweight face detection
  - Facenet512: produces 512-dim embeddings (same dimensionality as before)
  - Data augmentation: brightness/gamma/flip variants for lighting robustness
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

# DeepFace — lazy-loads the Facenet512 model on first call
from deepface import DeepFace

# ── Config ──────────────────────────────────────────────────────────────────
KNOWN_FACES_DIR  = "known_faces"
DB_OUTPUT_PATH   = "known_faces/face_embeddings_v6.pkl"
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# DeepFace settings
MODEL_NAME       = "Facenet512"      # 512-dim embeddings
DETECTOR_BACKEND = "opencv"          # Lightweight, fast on ARM/Pi

# Folders to skip (generic names, not a person)
SKIP_FOLDERS = {"Blind Stick Face Recog", "__pycache__"}


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
    Generate augmented versions of an image for lighting robustness.
    Returns list of BGR images (including original).
    """
    augmented = [img_bgr]

    # Horizontal flip
    augmented.append(cv2.flip(img_bgr, 1))

    # Gamma corrections (simulate different lighting)
    for gamma in [0.4, 0.6, 1.5, 2.0]:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in range(256)]).astype("uint8")
        augmented.append(cv2.LUT(img_bgr, table))

    # Brightness adjustments
    for alpha, beta in [(0.6, -20), (1.4, 20)]:
        augmented.append(cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta))

    # CLAHE on the luminance channel
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    augmented.append(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))

    return augmented


def extract_embedding(img_bgr):
    """
    Detect the largest face in a BGR image and return its 512-dim embedding.
    Returns (embedding_np, info_str) or (None, error_str).
    """
    try:
        # DeepFace.represent() handles detection + alignment + embedding in one call
        results = DeepFace.represent(
            img_path=img_bgr,           # Pass numpy array directly
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,     # Don't crash if no face found
            align=True,                  # Align face before embedding
        )

        if not results or len(results) == 0:
            return None, "no face detected"

        # If multiple faces found, pick the largest (by facial_area)
        best = max(results, key=lambda r: r.get("facial_area", {}).get("w", 0) *
                                           r.get("facial_area", {}).get("h", 0))

        embedding = np.array(best["embedding"], dtype=np.float32)

        # Verify we got a valid embedding
        if embedding.shape[0] != 512:
            return None, f"unexpected embedding dim: {embedding.shape[0]}"

        # Check if a face was actually detected (DeepFace with enforce_detection=False
        # may return an embedding of the whole image if no face is found)
        face_area = best.get("facial_area", {})
        face_w = face_area.get("w", 0)
        face_h = face_area.get("h", 0)
        img_h, img_w = img_bgr.shape[:2]

        # If the "face" is basically the entire image, it likely didn't detect a real face
        if face_w >= img_w * 0.95 and face_h >= img_h * 0.95:
            # Still accept it for training (the user provided a face photo),
            # but note it in the info
            pass

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        confidence = best.get("face_confidence", 0)
        return embedding, f"conf={confidence:.3f}, area={face_w}x{face_h}"

    except Exception as e:
        return None, f"error: {str(e)}"


def extract_embedding_with_fallback(img_bgr):
    """
    Try to extract an embedding, falling back to CLAHE enhancement for low-light images.
    """
    emb, info = extract_embedding(img_bgr)
    if emb is not None:
        return emb, info

    # Fallback: CLAHE enhanced image
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return extract_embedding(enhanced)


def train():
    print("=" * 62)
    print("  SMART BLIND STICK - FACE TRAINING v6 (DeepFace Facenet512)")
    print("=" * 62)
    print(f"  Model         : Facenet512 (512-dim embeddings)")
    print(f"  Detector      : OpenCV Haar Cascade")
    print(f"  Backend       : DeepFace (TF-Keras)")
    print(f"  Augmentation  : flip + gamma + brightness + CLAHE (9x)")
    print(f"  Source dir    : {os.path.abspath(KNOWN_FACES_DIR)}")
    print("=" * 62)

    if not os.path.exists(KNOWN_FACES_DIR):
        print(f"ERROR: '{KNOWN_FACES_DIR}' not found.")
        sys.exit(1)

    # Pre-warm DeepFace model (first call downloads + loads weights)
    print("\n  Loading Facenet512 model (first run downloads weights)...")
    try:
        # Create a small dummy image to trigger model loading
        dummy = np.zeros((160, 160, 3), dtype=np.uint8)
        dummy[40:120, 40:120, :] = 128  # Gray square
        DeepFace.represent(
            img_path=dummy,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
        )
        print("  Facenet512 model loaded successfully.\n")
    except Exception as e:
        print(f"  Warning: Model pre-warm returned: {e}")
        print("  Continuing — model will load on first real image.\n")

    print("Scanning images...")
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

    print(f"\n--- Extracting Facenet512 embeddings ---\n")
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
            emb, info = extract_embedding_with_fallback(aug_img)
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
        'model': 'Facenet512-DeepFace',
        'embedding_dim': 512,
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
