# -*- coding: utf-8 -*-
"""
Created on Fri May 22 18:35:55 2020

@author: Anustup
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

webapp = Flask(__name__)
model = pickle.load(open('weather.pkl', 'rb'))

@webapp.route('/')
def home():
    return render_template('index.html')

@webapp.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Predicted weather will be $ {}'.format(output))


if __name__ == "__main__":
    webapp.run(debug=True)