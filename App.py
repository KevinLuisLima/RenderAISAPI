#1. Objetivo - colocar o arquivo keras online para consulta do arduino
#2. URL base - localhost
#3. Endpoints - localhost/image_result (GET)
#4. Quais recursos -flask, tensorflow, opencv-python e numpy
from flask import Flask, jsonify, request
import os
import tensorflow as tf
import cv2
import numpy as np

App = Flask(__name__)
model = tf.keras.models.load_model('Identifica_Sala_Ocupada.keras')

def image_processor(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (448, 448))  # Ajuste conforme seu modelo
    img = img / 255.0  # Normalização, se necessário
    img = np.expand_dims(img, axis=0)
    return img

@App.route('/image_result',methods=['GET'])
def checkGivenImage():
    image_path = request.args.get('image_path')
    if not image_path or not os.path.exists(image_path):
        return jsonify({'error': 'Image not founds or invalid path || Imagem nao encontrada ou caminho invalido'}), 400
    try:
        img = image_processor(image_path)
        prediction = model.predict(img)

        if prediction[0][0] > 0.5:
            resultado = 'Pessoa na sala'
        else:
            resultado = 'Nenhuma pessoa na sala'

        return jsonify({'resultado': resultado})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

App.run(port=5000,host='localhost',debug=True)