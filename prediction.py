# Import các thư viện cần thiết
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import pickle

# Đọc dataset
data = pd.read_csv("cardio_train.csv", sep=';')

#Tạo copy của dataset để không ảnh hưởng tới dataset ban đầu
data_df = data.copy();

#Đổi tên cột cardio thành target
data_df = data_df.rename(columns={'cardio':'target'})
print(data_df.head())

#Mô hình hóa
# Phân chia dữ liệu thành đầu vào (X) và đầu ra (y). (y) chứa dữ liệu target và (x) chứa dữ liệu còn lại
X = data_df.drop(columns=['target', 'id'])
y = data_df['target']

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Khởi tạo mô hình RandomForest
model = LogisticRegression(max_iter=10000)
model.fit(X_train_scaled, y_train)

# Đánh giá mô hình
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác của mô hình: {accuracy * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
print(cm)

# Lưu mô hình và scaler vào file
pickle.dump(model, open('logistic_regression_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
