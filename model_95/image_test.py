# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import plaidml.keras as pk
pk.install_backend()
import matplotlib.pyplot as plt
from matplotlib.image import imread
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model


def cat_dog_classifier(image_file):
    img = load_img(image_file, target_size=(128,128))
    img = img_to_array(img)
    
    img = img.reshape(1,128,128,3)
    img = img.astype('float32')
    img = img * (1/255)
    
    model = load_model('cats_v_dogs_95.h5')
    
    result = model.predict(img)
    
    display_img = imread(image_file)
    plt.imshow(display_img)
    
    if result[0] > .5:
        print('\nIT\'S A DOG')
    else:
        print('\nIT\'S A CAT')


image_file = '~/Desktop/download.jpg'

cat_dog_classifier(image_file)
