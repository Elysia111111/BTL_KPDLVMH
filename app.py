from flask import Flask, render_template, request
import pickle
import numpy as np

# Load mô hình và scaler đã lưu
filename = 'logistic_regression_model.pkl'
model = pickle.load(open(filename, 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        # Lấy dữ liệu từ form
        age = int(request.form['age'])
        gender = request.form.get('gender')
        height = int(request.form['height'])
        weight = int(request.form['weight'])
        ap_hi = int(request.form['ap_hi'])
        ap_lo = int(request.form['ap_lo'])
        cholesterol = request.form.get('cholesterol')
        gluc = request.form.get('gluc')
        smoke = request.form.get('smoke')
        alco = request.form.get('alco')
        active = request.form.get('active')
        
        # Chuẩn hóa dữ liệu đầu vào
        data = np.array([[age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]])  # Điều chỉnh theo số lượng cột
        data_scaled = scaler.transform(data)

        # Dự đoán
        prediction = model.predict(data_scaled)

        # Hiển thị kết quả
        return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
