import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
data_dict = pickle.load(open(
    r'C:\Users\user\PyCharmMiscProject\DATA\pose_data.pickle',
    'rb'
))
data = np.array(data_dict['data'])
labels = np.array(data_dict['labels'])
print(f"Jumlah data: {len(data)}")
print(f"Distribusi label: Berdiri={sum(labels==0)}, Duduk={sum(labels==1)}")
x_train, x_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.1,
    shuffle=True,
    stratify=labels
)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)

print(f"\ndari sampel diklasifikasikan dengan benar!".format(score * 100))
print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_predict, target_names=['berdiri', 'duduk']))

with open(r'C:\Users\user\PyCharmMiscProject\DATA\pose_model.pickle', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Model berhasil disimpan!")
