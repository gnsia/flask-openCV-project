
import flask
from flask import Flask,render_template,url_for,request
import base64
import numpy as np
import cv2
from keras.models import load_model 
import io 
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
import json
import tensorflow as tf
from keras import backend as K



init_Base64 = 22   # data:image/png;base64, 로 시작하
app = Flask(__name__)

global model 
global sess
global graph

sess = tf.Session()       
graph = tf.get_default_graph() 
K.set_session(sess)

model = ResNet50(weights='imagenet')
json_data = json.load(open('imagenet_class_index.json'))

@app.route('/')
def home():
    return "ok~~~"


@app.route('/image', methods=['POST'])
def upload():       
    draw = request.form['photo_cap']
    draw = draw[init_Base64:]
    draw_decoded = base64.b64decode(draw)
    image = np.asarray(bytearray(draw_decoded), dtype="uint8")
    
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)    
    image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_AREA)                
    image = image.reshape(-1, 224, 224, 3)
    image = preprocess_input(image)
    
    pred = model.predict(image)
    r = json_data[f"{np.argmax(pred)}"][1]              
        
    return f"인식결과:{r}"
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8000)
