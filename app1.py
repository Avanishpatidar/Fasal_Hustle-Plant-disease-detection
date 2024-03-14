import os
import pathlib
import numpy as np
import tensorflow as tf
import keras
from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

# Load the model
model = load_model('model/v2.h5')

# Mapping of labels
labels ={'Apple___Apple_scab': 0, 'Apple___Black_rot': 1, 
        'Apple___Cedar_apple_rust': 2, 'Apple___healthy': 3, 'Blueberry___healthy': 4,
        'Cherry_(including_sour)___Powdery_mildew': 5, 'Cherry_(including_sour)___healthy': 6,  
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 7, 
        'Corn_(maize)___Common_rust_': 8, 'Corn_(maize)___Northern_Leaf_Blight': 9, 
        'Corn_(maize)___healthy': 10, 'Grape___Black_rot': 11, 
        'Grape___Esca_(Black_Measles)': 12, 
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 13, 
        'Grape___healthy': 14, 'Orange___Haunglongbing_(Citrus_greening)': 15, 
        'Peach___Bacterial_spot': 16, 'Peach___healthy': 17, 'Pepper,_bell___Bacterial_spot': 18, 
        'Pepper,_bell___healthy': 19, 'Potato___Early_blight': 20, 'Potato___Late_blight': 21,
        'Potato___healthy': 22, 'Raspberry___healthy': 23, 'Soybean___healthy': 24, 
        'Squash___Powdery_mildew': 25, 'Strawberry___Leaf_scorch': 26, 
        'Strawberry___healthy': 27, 'Tomato___Bacterial_spot': 28, 'Tomato___Early_blight': 29,
        'Tomato___Late_blight': 30, 'Tomato___Leaf_Mold': 31, 'Tomato___Septoria_leaf_spot': 32, 
        'Tomato___Spider_mites Two-spotted_spider_mite': 33, 'Tomato___Target_Spot': 34,
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 35, 
        'Tomato___Tomato_mosaic_virus': 36,
        'Tomato___healthy': 37}


def preprocess_image(file):
    path = 'F:/project/random_tests/'
    img = image.load_img(path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded_dims)

def get_prediction(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    result = np.argmax(predictions[0])
    return [k for k, v in labels.items() if v == result][0]

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', prediction="No file part")
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file without a filename.
        if file.filename == '':
            return render_template('index.html', prediction="No selected file")
        if file:
            # Create the directory if it doesn't exist
            save_dir = 'F:/project/random_tests/'
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'uploaded_image.jpg')
            img = Image.open(file)
            img.save(save_path)  # Save the uploaded image
            prediction = get_prediction("uploaded_image.jpg")
            return render_template('index.html', prediction=prediction)
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
