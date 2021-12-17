# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
#    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#import tensor as tensor
#from wtfml.engine import Engine
import os
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import joblib
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, SpatialDropout2D, BatchNormalization, Input, Activation, Dense, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.losses import binary_crossentropy
import pickle
import PIL
from PIL import Image
import flasgger
from flasgger import Swagger
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

#DATA_DIR = "C:/Users/PAGOU/Documents/R/Projet_Transverse/Datasets/DATAA"
DATA_DIR = "Datasets/DATAA"
model_path = "model.h5"
model = tf.keras.models.load_model(model_path)
#model._make_predict_function()
#dir_path = os.path.dirname(os.path.realpath(__file__))
#model = load_model(os.path.join(dir_path, MODEL_PATH))

#@app.route('/Templates/index', methods=['GET'])
#@app.route('/', methods=['GET'])
#def index():
 #   return render_template('index.html')
@app.route('/')
@app.route('/templates/index', methods=['GET'])
def get_root():
    print('sending root')
    return render_template('index.html')

@app.route('/uploadfile/docs')
#@app.route('/templates')
def get_docs():
    print('sending docs')
    return render_template('swaggerui.html')

def model_predict(image_path, model):
    img = image.load_img(image_path)
    test_image = img
    test_target = [0]
    # import tf.image as tfi #la librairie que Mr veut qu'on utilise
    #test_dirr = "/Users/PAGOU/Documents/R/Projet_Transverse/Datasets/0-with-mask.jpg"
    #img_path = test_dirr  # coming from API, should be a .png file
    # img_path = fullimg[550] #550 sans mask #exemple
    #pil_img = PIL.Image.open(test_image).convert('RGB')
    test_img = tf.keras.preprocessing.image.img_to_array(test_image)
    test_img = tf.image.resize(test_img, [120, 120], antialias=True)  # resize
    test_img = tf.expand_dims(test_img,axis=0)  # add a dimension with size of the batch, 1 in our case #0 si mask et 1 si pas de masque
    predictions = model.predict(test_img)
    return predictions

@app.route('/uploadfile', methods=['GET', 'POST'])
def upload_predict():

    #req = request.get_json()

    #data_train = req['data_train']
   # y_train = req['y_train']
    if request.method == 'POST':
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(
                DATA_DIR,
                image_file.filename
            )
            #dir_path = os.path.dirname(os.path.realpath(__file__))
            #model = load_model(os.path.join(dir_path, MODEL_PATH))
            image_file.save(image_location)
            preds = model_predict(image_location, model)[0]
            #pred = model.predict(image_location,model)[0]
            print(preds)
            #pred_class = decode_predictions(preds, top=1)
            #result = str(preds[0][0][1])
            #return result
            prediction = str(preds[0])
            #result1 = str(prediction)
            #result2 = str(prediction=0)
            return prediction
        return prediction

#@app.route('/', methods=['GET','POST'])
@app.route('/Templates/prediction', methods=['GET','POST'])
def upload_predictfile():

    #req = request.get_json()

    #data_train = req['data_train']
   # y_train = req['y_train']
    if request.method == 'POST':
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(
                DATA_DIR,
                image_file.filename
            )
            image_file.save(image_location)
            preds = model_predict(image_location, model)[0]
            #pred = model.predict(image_location,model)[0]
            print(preds)
            #pred_class = decode_predictions(preds, top=1)
            #result = str(preds[0][0][1])
            #return result

            return render_template("prediction.html", prediction= preds)
    return render_template("prediction.html", prediction=0)


    #    f = request.files['file']
   # prediction = model.predict(input_data_df)

    #pred_class = decode_predictions(prediction, top=1)
    #result = str(pred_class[0][0][1])
    #return result

"""
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST'
        f = request.files['file']
        """
"""def home():
    if request.method == 'POST':
        model = pickle.load(open('Projet_Transverse/filename.pkl', 'rb'))
        user_input = request.form.get('size')
        user_input = image(user_input)
        prediction = model.predict([[user_input]])
        print(prediction)
    return render_template('index.html')"""
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(debug=True)
#app.run(host='127.0.0.1', port=8000)
   # app.run(debug=True)
    #print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pychar
