import cv2
import threading
import time
import pygame
import os

DOORBELL_FILE = "doorbell.mp3"

# Video capture (default camera = 0)

cap = cv2.VideoCapture(0)

# Haarcascade file (downloaded with OpenCV)

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Debounce /state variables

sound_played = False
no_face_frames = 0
NO_FACE_RESET_FRAMES = 10   # 10 frames of consecutive sound play will be possible again if there is no face


# pygame mixer init (for mp3 playback)

pygame.mixer.init()

def ring_bell():
    """will be called on pythread so as not to block the main loop."""
    try:
        # Stop any currently playing sound and then play

        pygame.mixer.music.stop()
        pygame.mixer.music.load(DOORBELL_FILE)
        pygame.mixer.music.play()
    except Exception as e:
        print("Error playing sound:", e)

def main_loop():
    global sound_played, no_face_frames

    if not os.path.exists(DOORBELL_FILE):
        print(f"Error: '{DOORBELL_FILE}' not found in current folder.")
        return

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to get frame from camera.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detectMultiScale returns list of rectangles (faces)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

            if len(faces) > 0:
                # If the face is found, note it

                no_face_frames = 0
                if not sound_played:
                    # Playing sound in threads so the camera doesn't block the loop

                    t = threading.Thread(target=ring_bell, daemon=True)
                    t.start()
                    sound_played = True
                    print("Face detected — doorbell rung.")
                # (You can also draw a box if you wish)

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            else:
                # Increment counter if no face — for stable reset

                no_face_frames += 1
                if sound_played and no_face_frames >= NO_FACE_RESET_FRAMES:
                    sound_played = False
                    print("No faces for a while — ready to ring again.")

            cv2.imshow("Smart Doorbell", frame)

            # Press 'q' to exit

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Exiting.")

if __name__ == "__main__":
    main_loop()