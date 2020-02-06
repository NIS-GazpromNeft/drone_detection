import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import os

import cv2
import numpy as np

from tensorflow.keras import applications

#parameter setting
train_path = "save/"
brightness_change = 250
dropout_rate = 0.3 #in dense layer
number_of_epochs = 10
batch_size = 64

def load_images_from_folder(path):
    
    images = []
    
    for filename in os.listdir(path):
        img = cv2.imread(path + "/" + filename)
        images.append(img)
    
    return images

#load and merge all images 
#create vector Y
def data_loader(base_path):
    
    class_names = os.listdir(base_path)
    
    x_train = np.empty(shape=[0, 256, 256, 3], dtype = np.uint8)
    y_train = list()
    
    for name in class_names:
        new_x = np.reshape(load_images_from_folder(base_path + "/" + name), (-1,256,256,3))
        x_train = np.concatenate((x_train, new_x), axis = 0)
        y_train = np.concatenate((y_train, np.repeat(class_names.index(name), len(new_x))), axis = 0)
    
    return (x_train, y_train)

#random brightness increase/decrease
def change_brightness(img, left = -10, right = 10):    
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)    
    img_hsv[:,:,2] = img_hsv[:,:,2] + np.random.randint(low = left, high = right, size = 1)
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

#load data

x_train, y_train = data_loader(train_path)

#add random brightness
for index in range(x_train.shape[0]):
    x_train[index] = change_brightness(x_train[index], -brightness_change, brightness_change)

num_classes = len(np.unique(y_train))

base_model = applications.resnet50.ResNet50(weights= 'imagenet', include_top=False, input_shape= 
                                            (x_train.shape[1],x_train.shape[2],x_train.shape[3]))

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(dropout_rate)(x)
predictions = layers.Dense(num_classes, activation= 'softmax')(x)
model = keras.Model(inputs = base_model.input, outputs = predictions)
model.compile(optimizer= keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

callbacks = [
            keras.callbacks.TensorBoard(log_dir='./log/training_with_imagenet_weights', write_images=True,profile_batch = 100),
            ]

model.fit(x_train, y_train, epochs = number_of_epochs, batch_size = batch_size, validation_split = 0.3, callbacks = callbacks)


#test on real data
#test_path = ""
#x_test, y_test = data_loader(test_path)
#model.evaluate(x = x_test, y = y_test)