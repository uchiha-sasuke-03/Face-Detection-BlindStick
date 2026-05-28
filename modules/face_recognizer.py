"""
Face Recognizer Module (v7 - MobileFaceNet ONNX)
===================================================
Uses OpenCV DNN SSD for face detection and MobileFaceNet (ONNX Runtime)
for 128-dim embeddings. No PyTorch, no TensorFlow, no DeepFace required.

Key design choices:
  - OpenCV DNN SSD: fast, works in low-light
  - MobileFaceNet ONNX: 128-dim, ~1MB model, extremely fast on CPU
  - Cosine similarity threshold: 0.45 (tuned for 128-dim)
  - CLAHE fallback: enhances low-light images before re-trying detection
  - 5-second cooldown on name announcements
"""

import os, pickle, threading, time, cv2, numpy as np, urllib.request

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
SSD_PROTOTXT   = os.path.join(MODELS_DIR, "deploy.prototxt")
SSD_CAFFEMODEL = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
SSD_PROTOTXT_URL   = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
SSD_CAFFEMODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
MOBILEFACENET_PATH = os.path.join(MODELS_DIR, "mobilefacenet.onnx")
MOBILEFACENET_URL  = "https://huggingface.co/deepghs/insightface/resolve/main/buffalo_s/w600k_mbf.onnx?download=true"

SSD_CONFIDENCE = 0.5


def _ensure_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for path, url, desc in [
        (SSD_PROTOTXT, SSD_PROTOTXT_URL, "SSD prototxt"),
        (SSD_CAFFEMODEL, SSD_CAFFEMODEL_URL, "SSD caffemodel"),
        (MOBILEFACENET_PATH, MOBILEFACENET_URL, "MobileFaceNet ONNX"),
    ]:
        if not os.path.exists(path):
            print(f"  Downloading {desc}...")
            try:
                urllib.request.urlretrieve(url, path)
                print(f"  Done: {path}")
            except Exception as e:
                print(f"  FAILED: {desc}: {e}")
                return False
    return True


class FaceRecognizer:
    COSINE_THRESHOLD = 0.30
    DB_FILENAME      = "face_embeddings_v7.pkl"

    def __init__(self, known_faces_dir="known_faces"):
        self.known_faces_dir = known_faces_dir
        self.is_processing = False
        self.last_seen = ""
        self.last_seen_time = 0
        self.frames_since_last_seen = 0

        self._current_name = None
        self._current_name_time = 0
        self._name_lock = threading.Lock()

        self.database = {}
        self.db_loaded = False

        self._detector = None
        self._embedder = None
        self._models_ready = False

        if not os.path.exists(known_faces_dir):
            os.makedirs(known_faces_dir)

        self._load_database()
        if self.db_loaded:
            threading.Thread(target=self._load_models, daemon=True).start()

    def _load_models(self):
        """Load face detector and embedder models (background thread)."""
        try:
            if not _ensure_models():
                print("Face Model: Failed to download models. Recognition disabled.")
                return

            self._detector = cv2.dnn.readNetFromCaffe(SSD_PROTOTXT, SSD_CAFFEMODEL)

            import onnxruntime as ort
            session = ort.InferenceSession(MOBILEFACENET_PATH, providers=['CPUExecutionProvider'])
            self._embedder = session
            self._embed_input_name = session.get_inputs()[0].name

            # Warm up
            dummy = np.zeros((1, 3, 112, 112), dtype=np.float32)
            session.run(None, {self._embed_input_name: dummy})

            self._models_ready = True
            print("Face Model: MobileFaceNet + SSD loaded successfully")
        except Exception as e:
            print(f"Face Model: Load failed - {e}")

    def _load_database(self):
        db_path = os.path.join(self.known_faces_dir, self.DB_FILENAME)
        if not os.path.exists(db_path):
            print("Face DB: No trained database found. Run 'python train_faces.py' first.")
            return
        try:
            with open(db_path, 'rb') as f:
                data = pickle.load(f)
            self.database = data.get('people', {})
            model = data.get('model', 'unknown')
            trained_at = data.get('trained_at', 'unknown')
            total = sum(d['count'] for d in self.database.values())
            names = ', '.join(self.database.keys())
            print(f"Face DB: Loaded {len(self.database)} people ({total} embeddings) [{model}]")
            print(f"Face DB: Trained at {trained_at}")
            print(f"Face DB: Known people: {names}")
            self.db_loaded = True
        except Exception as e:
            print(f"Face DB: Failed to load - {e}")

    def _detect_faces(self, img_bgr):
        """Detect faces using OpenCV DNN SSD. Returns list of (x1,y1,x2,y2)."""
        if self._detector is None: return []
        h, w = img_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(img_bgr, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
        self._detector.setInput(blob)
        detections = self._detector.forward()
        faces = []
        for i in range(detections.shape[2]):
            if detections[0, 0, i, 2] < SSD_CONFIDENCE: continue
            x1 = max(0, int(detections[0, 0, i, 3] * w))
            y1 = max(0, int(detections[0, 0, i, 4] * h))
            x2 = min(w, int(detections[0, 0, i, 5] * w))
            y2 = min(h, int(detections[0, 0, i, 6] * h))
            if x2 > x1 and y2 > y1: faces.append((x1, y1, x2, y2))
        faces.sort(key=lambda f: (f[2]-f[0])*(f[3]-f[1]), reverse=True)
        return faces

    def _get_face_embedding(self, face_bgr):
        """Get 128-dim embedding from a cropped face image."""
        if self._embedder is None: return None
        face = cv2.resize(face_bgr, (112, 112))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = np.transpose(face, (2, 0, 1)).astype(np.float32)
        face = (face - 127.5) / 128.0
        inp = np.expand_dims(face, axis=0)
        out = self._embedder.run(None, {self._embed_input_name: inp})[0]
        emb = out.flatten().astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0: emb = emb / norm
        return emb

    @staticmethod
    def _cosine_sim(a, b):
        return float(np.dot(a, b))

    def _get_embedding(self, img_bgr):
        """Detect best face and return embedding. CLAHE fallback for low-light."""
        if not self._models_ready: return None

        emb = self._try_get_embedding(img_bgr)
        if emb is not None: return emb

        # CLAHE fallback
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8)).apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        return self._try_get_embedding(enhanced)

    def _try_get_embedding(self, img_bgr):
        try:
            faces = self._detect_faces(img_bgr)
            if not faces: return None
            x1, y1, x2, y2 = faces[0]
            crop = img_bgr[y1:y2, x1:x2]
            if crop.size == 0: return None
            return self._get_face_embedding(crop)
        except Exception:
            return None

    def _find_best_match(self, live_emb):
        best_name, best_score = None, -1.0
        for name, data in self.database.items():
            avg_score = self._cosine_sim(live_emb, data['average'])
            if avg_score > self.COSINE_THRESHOLD * 0.75:
                individual_scores = [self._cosine_sim(live_emb, emb) for emb in data['embeddings']]
                best_individual = max(individual_scores)
                combined = 0.40 * avg_score + 0.60 * best_individual
            else:
                combined = avg_score
            if combined > best_score:
                best_score = combined
                best_name = name
        if best_score >= self.COSINE_THRESHOLD:
            return best_name, best_score
        return None, 0

    def identify_person(self, frame, xyxy):
        """Called from YOLO loop when a 'person' is detected. Returns name or None."""
        if not self._models_ready or not self.db_loaded: return None
        try:
            x1 = max(0, int(xyxy[0]))
            y1 = max(0, int(xyxy[1]))
            x2 = min(frame.shape[1], int(xyxy[2]))
            y2 = min(frame.shape[0], int(xyxy[3]))
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0: return None
            emb = self._get_embedding(person_crop)
            if emb is None: return None
            name, score = self._find_best_match(emb)
            return name
        except Exception:
            return None

    def recognize(self, frame, tts_manager):
        """Standalone face recognition on the full frame (non-blocking)."""
        if not self._models_ready or self.is_processing or not self.db_loaded: return

        def _process():
            self.is_processing = True
            try:
                emb = self._get_embedding(frame)
                if emb is None: return
                name, score = self._find_best_match(emb)
                if name is None: return

                with self._name_lock:
                    self._current_name = name
                    self._current_name_time = time.time()

                now = time.time()
                if name != self.last_seen or (now - self.last_seen_time) > 5:
                    tts_manager.speak(f"I see {name}")
                    self.last_seen = name
                    self.last_seen_time = now
                    self.frames_since_last_seen = 0
            except Exception:
                pass
            finally:
                self.is_processing = False

        self.frames_since_last_seen += 1
        if self.frames_since_last_seen > 150: self.last_seen = ""
        threading.Thread(target=_process, daemon=True).start()

    def get_cached_name(self):
        with self._name_lock:
            if self._current_name and (time.time() - self._current_name_time) < 3:
                return self._current_name
        return None


face_rec = FaceRecognizer()
