#1. Objetivo - colocar o arquivo keras online para consulta do arduino
#2. URL base - localhost
#3. Endpoints - localhost/image_result (GET)
#4. Quais recursos - flask, tensorflow, opencv-python e numpy
from flask import Flask, jsonify, request
import os
import tensorflow as tf
import cv2
import numpy as np
import tempfile
import gdown
MODEL_ID = '1cObHXn3GtPz_WyTz3Jtkhx-Qu-BLnf8p'
MODEL_PATH = 'Identifica_Sala_Ocupada.keras'

App = Flask(__name__)

# Carrega o modelo remotamente
if not os.path.exists(MODEL_PATH):
    print("Baixando modelo...")
    gdown.download(id=MODEL_ID, output=MODEL_PATH, quiet=False)
    print("Modelo baixado com sucesso.")

model = tf.keras.models.load_model(MODEL_PATH)

def process_image_from_bytes(image_bytes):
    # Escreve bytes temporariamente para ler com OpenCV
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    img = cv2.imread(tmp_path)
    os.remove(tmp_path)  # remove o arquivo temporário

    if img is None:
        raise ValueError("Imagem inválida ou corrompida.")

    img = cv2.resize(img, (448, 448))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_from_array(image_bytes):
    img = process_image_from_bytes(image_bytes)
    prediction = model.predict(img)
    return prediction

@App.route('/image_result', methods=['POST'])
def checkGivenImage():
    if 'image' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Nome do arquivo vazio'}), 400

    try:
        # Lê a imagem como bytes e converte para array
        img_bytes = file.read()
        prediction = predict_from_array(img_bytes)

        if prediction[0][0] > 0.5:
            resultado = 'Pessoa detectada na sala'
        else:
            resultado = 'Nenhuma pessoa detectada na sala'

        return jsonify({'resultado': resultado})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    App.run(port=5000, host='localhost', debug=True)
    