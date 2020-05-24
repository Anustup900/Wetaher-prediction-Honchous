# -*- coding: utf-8 -*-
"""
Created on Fri May 22 18:35:55 2020

@author: Anustup
"""
#partly cloudy = 1
#mostly cloudy =2
#overcast=3
#foggy =4
#Breezy and 2=5
#clear=6
#Breezy and 3=7
#Humid and 2=8
#Windy and 4=9
#Humid and 1=10
#Windy and 3=11
#Breezy and 4=12
#Breezy and 1=13
#Windy and 1=14

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

webapp = Flask(__name__)
model = pickle.load(open('weather.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Predicted weather will be {}'.format(output))



if __name__ == "__main__":
    app.run(debug=True)
