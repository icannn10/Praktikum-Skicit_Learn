import os
import pickle
import mediapipe as mp
import cv2
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
DATA_DIR = r'C:\Users\PyCharmMiscProject\data_pose'
data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, dir_)
    for img_path in os.listdir(class_dir):
        img = cv2.imread(os.path.join(class_dir, img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        data_aux = []
        x_ = []
        y_ = []
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)
            for lm in results.pose_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            data.append(data_aux)
            labels.append(int(dir_))
with open('pose_data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
print("Ekstraksi selesai. File tersimpan sebagai pose_data.pickle")