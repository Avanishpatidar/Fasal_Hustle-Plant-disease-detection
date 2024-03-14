import os
import pathlib
import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model 
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
model =keras.models.load_model('model/v2.h5')

# json_config = model.to_json()
# with open('model_config.json', 'w') as json_file:
#     json_file.write(json_config)

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['acc'])
# print(model.summary())

labels={'Apple___Apple_scab': 0, 'Apple___Black_rot': 1, 
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
  img = image.load_img(path + file, target_size=(224,224))
  img_array = image.img_to_array(img)
  img_array_expanded_dims = np.expand_dims(img_array, axis=0)
  return tf.keras.applications.mobilenet_v2.preprocess_input(img_array_expanded_dims)


def get_prediction(image, level='all'):
  preprocessed_image = preprocess_image(image)
  predictions = model.predict(preprocessed_image)

  ind = np.argpartition(predictions[0], -5)[-5:]
  result = np.argmax(predictions[0])
  top5= predictions[0][ind]

  if level == 'single':
     for k,v in labels.items():
       if v == result: return k
  else:
    for k,v in labels.items():
      if v in np.sort(ind):
        idx = np.where(ind == v)[0]
        print(f'{top5[idx]} ~> {k}')



ans =get_prediction('download_2.jpeg')
print(ans)


