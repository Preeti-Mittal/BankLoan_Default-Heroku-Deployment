#!/usr/bin/env python
# coding: utf-8

import os
import flask
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
import pickle


app = Flask(__name__)

model = pickle.load(open("RF1_500.pkl","rb"))


@app.route('/')

def home():
    return render_template('index.html')
  
@app.route('/predict' , methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = prediction[0]
    
    if(prediction[0]==0):
        output = "not default"
    else:
        output = "default"
    return render_template('index.html', prediction_text = "This customer will {}".format(output))
    

if __name__ == "__main__":
    app.debug = True
    app.run()

