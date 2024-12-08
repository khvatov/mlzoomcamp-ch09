#!/usr/bin/env python
# coding: utf-8

# ## Alex Khvatov Homework 9


import tensorflow.lite as tflite
from io import BytesIO
from urllib import request
import numpy as np


interpreter = tflite.Interpreter(model_path='model_2024_hairstyle.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

from PIL import Image

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess_input(x):
    x/=127.5
    x -= 1.
    return x
    

#url = 'https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg'

def predict(url:str):
    img_raw = download_image(url)
    img = prepare_image(img_raw, (200,200))

    x = np.array(img, dtype='float32')
    X_raw = np.array([x])
    
    X = preprocess_input(X_raw)
    
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    preds = interpreter.get_tensor(output_index)

    return preds[0].tolist()[0]
    



