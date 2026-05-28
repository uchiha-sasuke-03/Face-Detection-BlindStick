"""
Face Embedding Training Script (v7 - MobileFaceNet ONNX)
==========================================================
Uses OpenCV DNN SSD for face detection and MobileFaceNet (ONNX) for
128-dimensional face embeddings. No PyTorch, no TensorFlow required.

Strategy:
  - OpenCV DNN SSD: fast, accurate face detection (works in low-light)
  - MobileFaceNet: 512-dim embeddings via ONNX Runtime (~14MB model)
  - Heavy augmentation: 21 variants per image (lighting, rotation, blur, noise, crop)
  - Stores per-person embeddings + averaged centroid for fast matching
  - Model stays lightweight — only training data diversity increases

Usage:
    python train_faces.py
"""

import os, sys, pickle, cv2, numpy as np, time, urllib.request

KNOWN_FACES_DIR  = "known_faces"
DB_OUTPUT_PATH   = "known_faces/face_embeddings_v7.pkl"
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
SKIP_FOLDERS = {"Blind Stick Face Recog", "__pycache__"}

MODELS_DIR = "models"
SSD_PROTOTXT   = os.path.join(MODELS_DIR, "deploy.prototxt")
SSD_CAFFEMODEL = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
SSD_PROTOTXT_URL   = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
SSD_CAFFEMODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

MOBILEFACENET_PATH = os.path.join(MODELS_DIR, "mobilefacenet.onnx")
MOBILEFACENET_URL  = "https://huggingface.co/deepghs/insightface/resolve/main/buffalo_s/w600k_mbf.onnx?download=true"

SSD_CONFIDENCE = 0.5


def ensure_models():
    """Download model files if they don't exist."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    downloads = [
        (SSD_PROTOTXT, SSD_PROTOTXT_URL, "SSD prototxt"),
        (SSD_CAFFEMODEL, SSD_CAFFEMODEL_URL, "SSD caffemodel (~10MB)"),
        (MOBILEFACENET_PATH, MOBILEFACENET_URL, "MobileFaceNet ONNX (~14MB)"),
    ]
    for path, url, desc in downloads:
        if not os.path.exists(path):
            print(f"  Downloading {desc}...")
            try:
                urllib.request.urlretrieve(url, path)
                print(f"  Done: {path}")
            except Exception as e:
                print(f"  FAILED to download {desc}: {e}")
                print(f"    URL: {url}")
                print(f"    Place file at: {os.path.abspath(path)}")
                sys.exit(1)


def collect_images(root_dir):
    entries = []
    for item in sorted(os.listdir(root_dir)):
        item_path = os.path.join(root_dir, item)
        if os.path.isfile(item_path) and item.lower().endswith(IMAGE_EXTENSIONS):
            entries.append((item_path, os.path.splitext(item)[0]))
        elif os.path.isdir(item_path):
            if item in SKIP_FOLDERS or item.startswith('.'): continue
            for fname in sorted(os.listdir(item_path)):
                fpath = os.path.join(item_path, fname)
                if os.path.isfile(fpath) and fname.lower().endswith(IMAGE_EXTENSIONS):
                    entries.append((fpath, item))
    return entries


def _rotate_image(img, angle):
    """Rotate image by a small angle, keeping the same size."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _add_noise(img, intensity=15):
    """Add Gaussian noise to simulate camera noise / low-light grain."""
    noise = np.random.normal(0, intensity, img.shape).astype(np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


def _center_crop(img, ratio=0.8):
    """Center-crop the image and resize back to original dimensions."""
    h, w = img.shape[:2]
    ch, cw = int(h * ratio), int(w * ratio)
    y1, x1 = (h - ch) // 2, (w - cw) // 2
    cropped = img[y1:y1+ch, x1:x1+cw]
    return cv2.resize(cropped, (w, h))


def augment_image(img_bgr):
    """
    Generate 21 augmented versions for maximum embedding diversity.
    All operations are lightweight CPU ops — no GPU needed.
    """
    augmented = [img_bgr]

    # 1. Horizontal flip (1 variant)
    augmented.append(cv2.flip(img_bgr, 1))

    # 2. Gamma corrections — simulate different lighting (5 variants)
    for gamma in [0.3, 0.5, 0.7, 1.5, 2.5]:
        table = np.array([((i / 255.0) ** (1.0/gamma)) * 255 for i in range(256)]).astype("uint8")
        augmented.append(cv2.LUT(img_bgr, table))

    # 3. Brightness/contrast adjustments (3 variants)
    for alpha, beta in [(0.5, -30), (0.8, -10), (1.5, 30)]:
        augmented.append(cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta))

    # 4. CLAHE enhancement — helps low-light (1 variant)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(l)
    augmented.append(cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR))

    # 5. Aggressive CLAHE — extreme low-light recovery (1 variant)
    lab2 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l2, a2, b2 = cv2.split(lab2)
    l2 = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4, 4)).apply(l2)
    augmented.append(cv2.cvtColor(cv2.merge([l2, a2, b2]), cv2.COLOR_LAB2BGR))

    # 6. Slight rotations — handle head tilt (4 variants)
    for angle in [-10, -5, 5, 10]:
        augmented.append(_rotate_image(img_bgr, angle))

    # 7. Gaussian blur — simulate motion/out-of-focus (1 variant)
    augmented.append(cv2.GaussianBlur(img_bgr, (5, 5), 0))

    # 8. Gaussian noise — simulate camera sensor noise (1 variant)
    augmented.append(_add_noise(img_bgr, intensity=15))

    # 9. Center crop + resize — simulate different distances (1 variant)
    augmented.append(_center_crop(img_bgr, ratio=0.75))

    # Total: 1 + 1 + 5 + 3 + 1 + 1 + 4 + 1 + 1 + 1 = 19 variants
    # (plus original = 20, but original is already counted)
    return augmented


class FaceDetectorSSD:
    def __init__(self, prototxt, caffemodel, confidence=0.5):
        self.net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        self.confidence = confidence

    def detect(self, img_bgr):
        h, w = img_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(img_bgr, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
        self.net.setInput(blob)
        detections = self.net.forward()
        faces = []
        for i in range(detections.shape[2]):
            if detections[0, 0, i, 2] < self.confidence: continue
            x1 = max(0, int(detections[0, 0, i, 3] * w))
            y1 = max(0, int(detections[0, 0, i, 4] * h))
            x2 = min(w, int(detections[0, 0, i, 5] * w))
            y2 = min(h, int(detections[0, 0, i, 6] * h))
            if x2 > x1 and y2 > y1:
                faces.append((x1, y1, x2, y2))
        faces.sort(key=lambda f: (f[2]-f[0]) * (f[3]-f[1]), reverse=True)
        return faces


class MobileFaceNetEmbedder:
    def __init__(self, model_path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        out_shape = self.session.get_outputs()[0].shape
        self.embedding_dim = out_shape[1] if len(out_shape) > 1 else 128

    def get_embedding(self, face_bgr):
        face = cv2.resize(face_bgr, (112, 112))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = np.transpose(face, (2, 0, 1)).astype(np.float32)
        face = (face - 127.5) / 128.0
        inp = np.expand_dims(face, axis=0)
        out = self.session.run(None, {self.input_name: inp})[0]
        emb = out.flatten().astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0: emb = emb / norm
        return emb


def extract_embedding(detector, embedder, img_bgr):
    faces = detector.detect(img_bgr)
    if not faces: return None, "no face detected"
    x1, y1, x2, y2 = faces[0]
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0: return None, "empty crop"
    return embedder.get_embedding(crop), f"area={x2-x1}x{y2-y1}"


def extract_embedding_with_fallback(detector, embedder, img_bgr):
    emb, info = extract_embedding(detector, embedder, img_bgr)
    if emb is not None: return emb, info
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8)).apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return extract_embedding(detector, embedder, enhanced)


def train():
    print("=" * 62)
    print("  SMART BLIND STICK - FACE TRAINING v7 (MobileFaceNet ONNX)")
    print("=" * 62)
    print(f"  Model         : MobileFaceNet (512-dim embeddings)")
    print(f"  Detector      : OpenCV DNN SSD")
    print(f"  Backend       : ONNX Runtime (CPU)")
    print(f"  Augmentation  : 19 variants (lighting+rotation+blur+noise+crop)")
    print(f"  Source dir    : {os.path.abspath(KNOWN_FACES_DIR)}")
    print("=" * 62)

    if not os.path.exists(KNOWN_FACES_DIR):
        print(f"ERROR: '{KNOWN_FACES_DIR}' not found."); sys.exit(1)

    print("\n  Checking model files...")
    ensure_models()

    print("\n  Loading face detector (OpenCV DNN SSD)...")
    detector = FaceDetectorSSD(SSD_PROTOTXT, SSD_CAFFEMODEL, SSD_CONFIDENCE)
    print("  Face detector loaded.")

    print("  Loading MobileFaceNet (ONNX Runtime)...")
    embedder = MobileFaceNetEmbedder(MOBILEFACENET_PATH)
    print(f"  MobileFaceNet loaded (dim: {embedder.embedding_dim}).")

    # Warm up
    dummy = np.zeros((160, 160, 3), dtype=np.uint8)
    dummy[40:120, 40:120, :] = 128
    embedder.get_embedding(dummy)
    print("  Model warmed up.\n")

    entries = collect_images(KNOWN_FACES_DIR)
    if not entries: print("ERROR: No images found."); sys.exit(1)

    people = {}
    for path, name in entries: people.setdefault(name, []).append(path)
    print(f"Found {len(entries)} images for {len(people)} people:")
    for name, paths in people.items(): print(f"  >> {name}: {len(paths)} image(s)")

    print(f"\n--- Extracting MobileFaceNet embeddings ---\n")
    database, success_count, fail_count = {}, 0, 0
    start_time = time.time()

    for i, (image_path, name) in enumerate(entries, 1):
        short = os.path.basename(image_path)
        if len(short) > 40: short = short[:37] + "..."
        print(f"  [{i:2d}/{len(entries)}] {name}/{short}")

        img = cv2.imread(image_path)
        if img is None: fail_count += 1; print("           SKIP - cannot read"); continue

        augmented_images = augment_image(img)
        img_success = 0
        for aug_img in augmented_images:
            emb, info = extract_embedding_with_fallback(detector, embedder, aug_img)
            if emb is not None: database.setdefault(name, []).append(emb); img_success += 1

        if img_success > 0:
            success_count += 1
            print(f"           OK - {img_success}/{len(augmented_images)} variants embedded")
        else:
            fail_count += 1; print("           SKIP - no face in any variant")

    elapsed = time.time() - start_time
    if not database: print("\nERROR: No faces extracted."); sys.exit(1)

    print(f"\nComputing per-person averaged embeddings...")
    final_db = {}
    for name, emb_list in database.items():
        avg = np.mean(emb_list, axis=0).astype(np.float32)
        norm = np.linalg.norm(avg)
        if norm > 0: avg = avg / norm
        final_db[name] = {'embeddings': emb_list, 'average': avg, 'count': len(emb_list)}

    output = {
        'model': 'MobileFaceNet-ONNX',
        'embedding_dim': embedder.embedding_dim,
        'people': final_db,
        'trained_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(DB_OUTPUT_PATH, 'wb') as f: pickle.dump(output, f)

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
    for name, data in final_db.items(): print(f"  [OK] {name}: {data['count']} embedding(s)")
    print(f"\nRestart main.py to use the new face database.")


if __name__ == '__main__':
    train()
