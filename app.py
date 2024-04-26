from __future__ import division, print_function
import os
from flask import Flask
from tensorflow.keras.models import load_model
import requests
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import sys
import os
import glob
import re
import numpy as np

app = Flask(__name__)

MODEL_URL = 'https://storage.googleapis.com/all-model-bucket/all_model.h5'
MODEL_PATH = 'all_model.h5'

def download_model(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)

# Download model file from cloud storage
download_model(MODEL_URL, MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(220, 220))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x = np.expand_dims(x, axis=0)




    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="The Condition is Benign"
    elif preds==1:
        preds="The Condition is Early"
    elif preds==1:
        preds="The Condition is Pre"
    else:
        preds="The Condition is Pro"


    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
