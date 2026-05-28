import cv2
import time
from ultralytics import YOLO

# Import our custom modules
from modules.audio_tts import tts
from modules.spatial_grid import analyze_position
from modules.ocr_reader import ocr
from modules.face_recognizer import face_rec
from modules.traffic_analyzer import traffic_analyzer
from modules.road_signs import road_sign_detector

def main():
    print("Initializing Advanced Blind Stick System...")
    tts.speak("System starting up")
    
    # Switch to YOLOv8 Nano (yolov8n) for massive performance boost on CPU/RPi
    model = YOLO('yolov8n.pt')
    
    cap = cv2.VideoCapture(0)
    # Lower resolution to speed up frame reading and rendering
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # To avoid speaking the same object constantly
    last_spoken_objects = {}
    last_results = [] # Cache results for frame skipping
    
    frame_count = 0

    print("System ready. Press 'q' to quit, 'r' to read text.")
    tts.speak("System ready.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_height, frame_width = frame.shape[:2]
        frame_count += 1

        # 1. Run YOLO Object Detection (skip every other frame to boost FPS)
        if frame_count % 2 == 0:
            results = model.predict(frame, verbose=False, conf=0.5)
            last_results = results[0].boxes if results else []
        
        current_objects = []

        for box in last_results:
            # Get class name
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
                
            # Get coordinates
            xyxy = box.xyxy[0].cpu().numpy()
            
            # 2. Grid & Spatial Awareness
            location, distance = analyze_position(xyxy, frame_width, frame_height)
            
            object_id = f"{class_name}_{location}_{distance}"
            current_objects.append(object_id)
            
            # Draw bounding box for visual debugging (always, regardless of distance)
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {distance} {location}", (int(xyxy[0]), int(xyxy[1])-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Only speak when object is CLOSE
            if distance != "close":
                continue

            if class_name not in ["traffic light", "stop sign"]:
                # If YOLO detects a "person", try face recognition to get their name
                spoken_name = class_name
                if class_name == "person":
                    recognized = face_rec.identify_person(frame, xyxy)
                    if recognized:
                        spoken_name = recognized
                    else:
                        cached = face_rec.get_cached_name()
                        if cached:
                            spoken_name = cached

                # Recognized faces get IMMEDIATE priority announcement (2s cooldown)
                if spoken_name != "person" and spoken_name != class_name:
                    face_id = f"face_{spoken_name}"
                    if face_id not in last_spoken_objects or (time.time() - last_spoken_objects[face_id]) > 2:
                        tts.speak(f"I see {spoken_name}")
                        last_spoken_objects[face_id] = time.time()
                else:
                    # Generic objects/unrecognized persons use normal 3s cooldown
                    obj_id = f"{spoken_name}_{location}_{distance}"
                    if obj_id not in last_spoken_objects or (time.time() - last_spoken_objects[obj_id]) > 3:
                        tts.speak(f"{spoken_name} {distance}, {location}")
                        last_spoken_objects[obj_id] = time.time()
            else:
                # Traffic lights and stop signs — only when close
                if class_name == "traffic light":
                    cropped = frame[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                    if cropped.size > 0:
                        traffic_analyzer.analyze_and_speak(cropped, tts)
                elif class_name == "stop sign":
                    cropped = frame[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                    if cropped.size > 0:
                        road_sign_detector.analyze_sign(cropped, ocr, tts)

        # Clean up old spoken objects memory
        current_time = time.time()
        keys_to_delete = [k for k, v in last_spoken_objects.items() if (current_time - v) > 10]
        for k in keys_to_delete:
            del last_spoken_objects[k]

        # Face recognition is now integrated into the YOLO loop via identify_person()
        # No standalone recognize() call needed — avoids duplicate announcements

        # Draw UI
        cv2.putText(frame, "Press 'r' to read text, 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Blind Stick View", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # 4. OCR Trigger
        elif key == ord('r'):
            tts.speak("Scanning for text")
            ocr.read_text(frame, tts)

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    tts.stop()

if __name__ == "__main__":
    main()
