#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

from flask import Flask, render_template
import tensorflow as tf
from keras.models import load_model
import numpy as np
import re
from flask import request
import json
import math

# Force CPU
num_cores = 1
num_CPU = 1
num_GPU = 0
config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=num_cores,
        inter_op_parallelism_threads=num_cores,
        allow_soft_placement=True,
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU}
        )
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
# Force CPU


# Utils
sampleLength = 400
def generateChars(data, pattern, code=False, position=False):
  array = []
  tmp = []
  for line in data:
    line = ' ' + line
    for char in line:
      tmp_char = char.lower()
      match = pattern.findall(tmp_char)
      if len(tmp_char) != 1 or len(match) < 1:
        continue
      if len(tmp) < sampleLength:
        if code:
          if position:
            if tmp_char in alphabetObject:
              tmp.append(float(np.where(alphabet == tmp_char)[0][0]))
          else:
            tmp.append(float(ord(tmp_char)))
        else:
          tmp.append(tmp_char)
      else:
        array.append(tmp)
        tmp = []
        if code:
          if position:
            if tmp_char in alphabetObject:
              tmp.append(float(np.where(alphabet == tmp_char)[0][0]))
          else:
            tmp.append(float(ord(tmp_char)))
        else:
          tmp.append(tmp_char)
  return np.array(array)

def textToCharCode(text):
  return  np.array([generateChars(text, re.compile(r"(.*)"), True)[0]])
# Utils

app = Flask(__name__)
model = tf.keras.models.load_model('./models/conv_model.h5')

# Routing
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    classes = ['ru', 'de', 'en', 'ukr']
    ret = {}
    req_data = request.get_json()
    app.logger.info(req_data['data'])
    
    toPredict = req_data['data'].strip() + ' '
    count = math.ceil(400 / len(toPredict))
    toPredict = toPredict * count
    predictions = model.predict(np.array(textToCharCode(toPredict)).reshape(1, 20,20,1))
    ret['cnn'] = {
        'prediction': classes[np.argmax(predictions[0])]
    }
    for index, _class in enumerate(classes):
        ret['cnn'][_class] = str(predictions[0][index])

    return json.dumps(ret)
#Routing

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)