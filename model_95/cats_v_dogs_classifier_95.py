#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 11:28:12 2019

@author: nigelstory
"""

# building the CNN
import sys
import plaidml.keras as pk
pk.install_backend()
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

from datetime import datetime

s = datetime.now()

# Initialize CNN

classifier = Sequential()

# Adding layers

# convolution
classifier.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(128,128,3)))
classifier.add(BatchNormalization(axis=-1))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# 2nd convolution
classifier.add(Conv2D(filters=128, kernel_size=(3, 3)))
classifier.add(BatchNormalization(axis=-1))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# 3rd convolution
classifier.add(Conv2D(filters=256, kernel_size=(3, 3)))
classifier.add(BatchNormalization(axis=-1))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# 4th convolution
classifier.add(Conv2D(filters=256, kernel_size=(3, 3)))
classifier.add(BatchNormalization(axis=-1))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# 5th convolution
classifier.add(Conv2D(filters=512, kernel_size=(3, 3)))
classifier.add(BatchNormalization(axis=-1))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# flatten
classifier.add(Flatten())

# full connection w/ dropout  
classifier.add(Dense(units=1024))
classifier.add(Activation('relu'))
classifier.add(Dropout(0.5))

classifier.add(Dense(units=256))
classifier.add(Activation('relu')) 
classifier.add(Dropout(0.5))

# output layer
classifier.add(Dense(units=1))
classifier.add(Activation('sigmoid'))

# adjust learning rate for plateaus
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, 
                                            verbose=2, factor=0.5, min_lr=0.00001)

# save best model & early stopping
best_model = ModelCheckpoint('cats_v_dogs.h5', monitor='val_acc', verbose=2, 
                             save_best_only=True, mode='max')

early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-10, 
                               patience=25,restore_best_weights=True)

# compile
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Image Processing
train_datagen = ImageDataGenerator(rescale=1.0/255,
                                   zoom_range=0.2,
                                   height_shift_range=.1,
                                   width_shift_range=.1,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(128, 128), 
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(128, 128), 
                                            batch_size=32,
                                            class_mode='binary')

model = classifier.fit_generator(training_set,
                         steps_per_epoch=8000/32,
                         epochs=100,
                         validation_data=test_set,
                         validation_steps=2000,
                         verbose=1,
                         callbacks=[learning_rate_reduction,best_model,early_stopping]
                         )

print(f"\nElapsed time: {datetime.now() - s}")

import matplotlib.pyplot as plt
plt.subplot(211)
plt.title('Cross Entropy Loss')
plt.plot(model.history['loss'], color='blue', label='train')
plt.plot(model.history['val_loss'], color='orange', label='test')
plt.xticks([])
plt.legend()
# plot accuracy
plt.subplot(212)
plt.title('Classification Accuracy')
plt.plot(model.history['acc'], color='blue', label='train')
plt.plot(model.history['val_acc'], color='orange', label='test')
plt.legend()

plt.xticks([])
filename = sys.argv[0].split('/')[-1]
plt.savefig(filename + '_plot.png')
plt.show()


# save final model
model_json = classifier.to_json()
with open('cats_v_dog_model.json', 'w') as f:
    f.write(model_json)
