import pickle
import cv2
import mediapipe as mp
import numpy as np
model_dict = pickle.load(open('pose_model.pickle', 'rb'))
model = model_dict['model']
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()
labels_dict = {0: 'Berdiri', 1: 'Duduk'}
cap = cv2.VideoCapture(0)
while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for lm in results.pose_landmarks.landmark:
            x_.append(lm.x)
            y_.append(lm.y)
        for lm in results.pose_landmarks.landmark:
            data_aux.append(lm.x - min(x_))
            data_aux.append(lm.y - min(y_))
        prediction = model.predict([np.asarray(data_aux)])
        predicted_label = labels_dict[int(prediction[0])]
        cv2.putText(frame, predicted_label,
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.3, (0, 255, 0), 3)
    cv2.imshow("Pose Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()