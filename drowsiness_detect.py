import cv2
import numpy as np
import time
import pygame

# Initialize sound with full path
pygame.mixer.init()
ALERT_PATH = r'C:\Users\hp\Desktop\Driver-Drowsiness-Detector-master\audio\alert.wav'
pygame.mixer.music.load(ALERT_PATH)

# Test the sound at start
print("[DEBUG] Testing alert sound...")
pygame.mixer.music.play()
time.sleep(2)
pygame.mixer.music.stop()

# Load Haar cascade models
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

# Start video capture
cap = cv2.VideoCapture(0)
time.sleep(2)

# Counter to track closed eyes
closed_counter = 0
CLOSED_THRESHOLD = 30  # Adjust this to change drowsiness sensitivity

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        continue

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect face
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        print("[DEBUG] No face detected.")
        closed_counter = 0
        pygame.mixer.music.stop()

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
        print(f"[DEBUG] Eyes detected: {len(eyes)}")

        if len(eyes) == 0:
            closed_counter += 1
        else:
            closed_counter = 0

        # Draw eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # If eyes closed too long
        if closed_counter >= CLOSED_THRESHOLD:
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play(-1)
            cv2.putText(frame, "DROWSY!", (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        else:
            pygame.mixer.music.stop()

    cv2.imshow("Drowsiness Detector (Simple)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
