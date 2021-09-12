from flask import Flask, jsonify,  request, render_template
import numpy as np
import pickle
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from model import recommendation
from model import sentiment

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/recommend", methods=['POST'])
def predict():
    if (request.method == 'POST'):
        user_input = request.form['Username']
        output = sentiment(user_input)
        if(len(output) == 5):
            return render_template('index.html', prediction_text=output,post ='True')
        else :
            return render_template('index.html', prediction_text="Sorry, we cannot find any such user in the database!!",post ='False')
    else :
        return render_template('index.html')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
