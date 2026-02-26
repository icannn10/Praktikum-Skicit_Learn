import os
from time import sleep
import cv2
DATA_DIR = r'C:\Users\user\PyCharmMiscProject\DATA'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
number_of_classes = 2
dataset_size = 100
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
        cv2.imshow("frame", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    sleep(1)
    print("[INFO] Mulai pengambilan gambar...")
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow("frame", frame)
        img_path = os.path.join(class_folder, f"{counter}.jpg")
        cv2.imwrite(img_path, frame)
        counter += 1
        print(f"Saved: {img_path}")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Pengambilan data dihentikan.")
            break

print("[INFO] Selesai!")
cap.release()
cv2.destroyAllWindows()