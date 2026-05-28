import threading
import cv2
import numpy as np

class OCRReader:
    def __init__(self):
        self.is_reading = False
        self.last_text = ""

    def read_text(self, frame, tts_manager):
        """
        Reads text from the frame using Tesseract OCR and speaks it out loud.
        This runs in a thread to prevent blocking the video feed.
        """
        if self.is_reading:
            return

        def _process():
            self.is_reading = True
            try:
                import pytesseract

                # Auto-detect Tesseract path on Windows
                import os, platform
                if platform.system() == "Windows":
                    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
                    if os.path.exists(tesseract_path):
                        pytesseract.pytesseract.tesseract_cmd = tesseract_path

                # Preprocess for better OCR accuracy
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # CLAHE for contrast enhancement (helps in low-light)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)

                # Otsu thresholding for binarization
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Run Tesseract OCR
                detected_text = pytesseract.image_to_string(thresh, config='--psm 6').strip()

                if detected_text and detected_text != self.last_text:
                    tts_manager.speak(f"Text says: {detected_text}")
                    self.last_text = detected_text
                elif not detected_text:
                    tts_manager.speak("No text detected")
            except ImportError:
                tts_manager.speak("Tesseract OCR not installed. Please install pytesseract.")
                print("OCR Error: pytesseract not installed. Run: pip install pytesseract")
                print("Also install Tesseract engine: https://github.com/UB-Mannheim/tesseract/wiki")
            except Exception as e:
                print(f"OCR Error: {e}")
            finally:
                self.is_reading = False

        threading.Thread(target=_process, daemon=True).start()

ocr = OCRReader()
