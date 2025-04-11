import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, jsonify
import normalized

with open('./models/model.pkl','rb') as pkl_file:
    model = pickle.load(pkl_file)

app = Flask(__name__)

@app.route('/')
def index():
    msg = 'Проверка связи. Сервер запущен!'
    return msg

@app.route('/predict', methods=['POST'])
def get_predict():
    data = pd.read_json(request.json, dtype={str})
    data = normalized.get_normalized_data(data)
    y_pred = np.exp(model.predict(data))
    return jsonify({'prediction':y_pred.to_list()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)