import os
from time import sleep
import cv2
import mediapipe as mp
DATA_DIR = r'C:\Users\user\PyCharmMiscProject\DATA'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 2
dataset_size = 100
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)
for j in range(number_of_classes):
    class_folder = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)

    print(f"[INFO] Pengambilan data untuk class {j}")
    print("[INFO] Tekan 'Q' untuk mulai mengambil gambar")
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press Q', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.imshow("Pose Capture", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    sleep(1)
    print("[INFO] Mulai pengambilan gambar...")
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            continue
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )
            img_path = os.path.join(class_folder, f"{counter}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"Saved: {img_path}")
            counter += 1
        cv2.imshow("Pose Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Pengambilan data dihentikan.")
            break

print("[INFO] Selesai!")
cap.release()
cv2.destroyAllWindows()