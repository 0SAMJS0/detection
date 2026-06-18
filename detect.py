import math
import cv2
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import mediapipe as mp

print("Loading the face model, first run downloads it...")

model_path = hf_hub_download(
    repo_id="arnabdhar/YOLOv8-Face-Detection",
    filename="model.pt"
)

face_model = YOLO(model_path)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def count_fingers(hand_landmarks):
    lm = hand_landmarks.landmark
    count = 0

    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    for tip, pip in zip(finger_tips, finger_pips):
        if lm[tip].y < lm[pip].y:
            count += 1

    def distance(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    pinky_base = lm[17]
    thumb_tip = lm[4]
    thumb_base = lm[2]

    if distance(thumb_tip, pinky_base) > distance(thumb_base, pinky_base):
        count += 1

    return count


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open your camera. Another app may be using it.")


with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        face_results = face_model(frame, verbose=False)

        for box in face_results[0].boxes:
            confidence = float(box.conf[0])

            if confidence < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                "My Face",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb)

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                fingers = count_fingers(hand_landmarks)

                xs = [int(p.x * w) for p in hand_landmarks.landmark]
                ys = [int(p.y * h) for p in hand_landmarks.landmark]

                hx1, hy1 = max(min(xs) - 20, 0), max(min(ys) - 20, 0)
                hx2, hy2 = min(max(xs) + 20, w), min(max(ys) + 20, h)

                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    f"Hand: {fingers} fingers",
                    (hx1, hy1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2
                )

        cv2.putText(
            frame,
            "Press 'q' to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("Face + Fingers", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()