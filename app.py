import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn import preprocessing
import pickle

app = Flask(__name__)
xgb_model = pickle.load(open('xgb_model.pkl', 'rb'))
stacked_model = pickle.load(open('stacked_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    feature_list = request.form.to_dict()
    feature_list = list(feature_list.values())
    feature_list = list(map(int, feature_list))
    final_features = np.array(feature_list).reshape(1, 12) 
    
    prediction = 0.725 * xgb_model.predict(final_features) + 0.275 * stacked_model.predict(final_features)
    output = float(prediction[0])
    
    return render_template('index.html', prediction_text='Time will be {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)