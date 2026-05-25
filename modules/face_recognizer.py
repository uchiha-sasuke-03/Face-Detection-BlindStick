"""
Face Recognizer Module - Haar Cascade + face_recognition (v6)
==============================================================
Uses OpenCV Haar Cascade for face detection and dlib (via face_recognition)
for 128-dimensional embeddings. Lightweight enough for Raspberry Pi 4.

Key design choices:
  - Haar Cascade: fast detection, ships with OpenCV, ~30 FPS on Pi 4
  - face_recognition (dlib): 128-dim embeddings, excellent accuracy
  - Euclidean distance threshold: 0.50 (well-calibrated for 128-dim space)
  - 3-pass detection: original → CLAHE → histogram equalized
  - identify_person(): integrates directly with YOLO "person" detection
  - 5-second cooldown on name announcements
"""

import os
import pickle
import threading
import time
import cv2
import numpy as np

# Try to import face_recognition
try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except ImportError:
    FACE_REC_AVAILABLE = False
    print("Warning: face_recognition not installed. Facial recognition disabled.")
    print("  Install with: pip install face_recognition")

# Haar Cascade path (ships with OpenCV)
HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


class FaceRecognizer:
    # ── Configuration ───────────────────────────────────────────────────
    DISTANCE_THRESHOLD   = 0.50       # Maximum Euclidean distance for a match
    DB_FILENAME          = "face_embeddings_v6.pkl"

    def __init__(self, known_faces_dir="known_faces"):
        self.known_faces_dir = known_faces_dir
        self.is_processing = False
        self.last_seen = ""
        self.last_seen_time = 0
        self.frames_since_last_seen = 0

        # Cached name for fast lookups from YOLO loop
        self._current_name = None
        self._current_name_time = 0
        self._name_lock = threading.Lock()

        # Models
        self.cascade = None
        self.database = {}
        self.db_loaded = False

        if not FACE_REC_AVAILABLE:
            return

        if not os.path.exists(known_faces_dir):
            os.makedirs(known_faces_dir)

        self._load_detector()
        self._load_database()

    def _load_detector(self):
        """Load the Haar Cascade face detector."""
        try:
            self.cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
            if self.cascade.empty():
                print("Face Detector: Failed to load Haar Cascade")
                self.cascade = None
            else:
                print("Face Detector: Haar Cascade loaded successfully")
        except Exception as e:
            print(f"Face Detector: Failed to load - {e}")
            self.cascade = None

    def _load_database(self):
        """Load pre-computed face embeddings."""
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

    def _detect_faces_haar(self, img_bgr):
        """Detect faces using Haar Cascade. Returns list of (x, y, w, h)."""
        if self.cascade is None:
            return []

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces if len(faces) > 0 else []

    @staticmethod
    def _enhance_clahe(img_bgr):
        """Apply CLAHE enhancement for low-light images."""
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _enhance_histeq(img_bgr):
        """Apply histogram equalization for contrast improvement."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(gray)
        return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _euclidean_dist(a, b):
        """Euclidean distance between two vectors."""
        return float(np.linalg.norm(a - b))

    def _get_embedding(self, img_bgr):
        """
        Detect the best face in a BGR image and return its 128-dim embedding.
        Uses 3-pass detection: original → CLAHE → histogram equalization.
        Returns 128-dim numpy array or None.
        """
        if self.cascade is None:
            return None

        # Try original image first
        emb = self._try_get_embedding(img_bgr)
        if emb is not None:
            return emb

        # Fallback 1: CLAHE enhancement for low light
        enhanced = self._enhance_clahe(img_bgr)
        emb = self._try_get_embedding(enhanced)
        if emb is not None:
            return emb

        # Fallback 2: Histogram equalization
        histeq = self._enhance_histeq(img_bgr)
        return self._try_get_embedding(histeq)

    def _try_get_embedding(self, img_bgr):
        """Try to detect a face and get its embedding. Returns numpy array or None."""
        try:
            faces = self._detect_faces_haar(img_bgr)
            if len(faces) == 0:
                return None

            # Pick the largest face
            largest = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest

            # Convert Haar (x, y, w, h) to face_recognition format (top, right, bottom, left)
            face_location = (y, x + w, y + h, x)

            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Get 128-dim encoding (1 jitter for speed during live recognition)
            encodings = face_recognition.face_encodings(
                img_rgb,
                known_face_locations=[face_location],
                num_jitters=1,
                model="large"
            )

            if len(encodings) == 0:
                return None

            emb_np = encodings[0].astype(np.float32)

            # L2 normalize
            norm = np.linalg.norm(emb_np)
            if norm > 0:
                emb_np = emb_np / norm

            return emb_np

        except Exception:
            return None

    def _find_best_match(self, live_emb):
        """
        Compare a live embedding against the database.
        Returns (name, distance) or (None, float('inf')).

        Strategy:
          - Check average embedding first (fast rejection)
          - If promising, check individual embeddings
          - Score = 40% average_dist + 60% best_individual_dist
        """
        best_name = None
        best_dist = float('inf')

        for name, data in self.database.items():
            avg_dist = self._euclidean_dist(live_emb, data['average'])

            # If average is even remotely close, check individuals
            if avg_dist < self.DISTANCE_THRESHOLD * 1.5:
                individual_dists = [
                    self._euclidean_dist(live_emb, emb)
                    for emb in data['embeddings']
                ]
                best_individual = min(individual_dists)
                combined = 0.40 * avg_dist + 0.60 * best_individual
            else:
                combined = avg_dist

            if combined < best_dist:
                best_dist = combined
                best_name = name

        if best_dist <= self.DISTANCE_THRESHOLD:
            return best_name, best_dist

        return None, float('inf')

    def identify_person(self, frame, xyxy):
        """
        Called from YOLO loop when a "person" is detected.
        Crops the person region, detects a face, and matches it.

        Args:
            frame: Full BGR frame
            xyxy: YOLO bounding box [x1, y1, x2, y2]

        Returns:
            Person's name (str) if recognized, or None.
        """
        if not FACE_REC_AVAILABLE or not self.db_loaded:
            return None

        try:
            x1 = max(0, int(xyxy[0]))
            y1 = max(0, int(xyxy[1]))
            x2 = min(frame.shape[1], int(xyxy[2]))
            y2 = min(frame.shape[0], int(xyxy[3]))

            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                return None

            # Get face embedding from the person crop
            emb = self._get_embedding(person_crop)
            if emb is None:
                return None

            # Match against database
            name, dist = self._find_best_match(emb)
            return name

        except Exception:
            return None

    def recognize(self, frame, tts_manager):
        """
        Standalone face recognition on the full frame.
        Non-blocking: spawns a background thread.
        """
        if not FACE_REC_AVAILABLE or self.is_processing or not self.db_loaded:
            return

        def _process():
            self.is_processing = True
            try:
                emb = self._get_embedding(frame)
                if emb is None:
                    return

                name, dist = self._find_best_match(emb)
                if name is None:
                    return

                # Cache the result
                with self._name_lock:
                    self._current_name = name
                    self._current_name_time = time.time()

                # Speak the name
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
        if self.frames_since_last_seen > 150:
            self.last_seen = ""

        threading.Thread(target=_process, daemon=True).start()

    def get_cached_name(self):
        """Returns the most recently recognized name if fresh (< 3 seconds)."""
        with self._name_lock:
            if self._current_name and (time.time() - self._current_name_time) < 3:
                return self._current_name
        return None


face_rec = FaceRecognizer()
