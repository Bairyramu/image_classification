from __future__ import division, print_function
# coding=utf-8
#import sys
import os
#import glob
#import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

 
# load model
model_98 = load_model(r'new_model_98.h5')


def model_predict(img_path):
    img = image.load_img(img_path, target_size=(255, 255))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model_98.predict(x)
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
        preds = model_predict(file_path)
        
        score_list=list(preds[0])
        label=score_list.index(max(score_list))
		
        if label==0:
            result="It's a bike"
        elif label==1:
            result="It's a car"
        elif label==2:
            result="It's a laptop"
        else:
            result="It's a mobile"  		
        return result
    return None


if __name__ == '__main__':
    app.run()
