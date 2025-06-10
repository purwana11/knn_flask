# knn_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load dataset
df = pd.read_csv("Iris.csv")

# 2. Hapus kolom ID jika ada
if 'Id' in df.columns:
    df = df.drop(columns='Id')

# 3. Pisahkan fitur dan label
X = df.drop(columns='Species')
y = df['Species']

# 4. Normalisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 6. Buat model KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 7. Evaluasi akurasi
y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Akurasi: {acc:.2f}")

# 8. Simpan model dan scaler
joblib.dump(knn, 'model_knn.pkl')
joblib.dump(scaler, 'scaler.pkl')
