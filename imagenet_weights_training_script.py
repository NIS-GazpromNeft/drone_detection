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
    
    x_train0 = load_images_from_folder(base_path + "/empty")
    x_train1 = load_images_from_folder(base_path + "/box")
    x_train2 = load_images_from_folder(base_path + "/recorder")
    x_train3 = load_images_from_folder(base_path + "/box_recorder")
    x_train4 = load_images_from_folder(base_path + "/box_sonda")
    x_train5 = load_images_from_folder(base_path + "/recorder_sonda")
    x_train6 = load_images_from_folder(base_path + "/all")
    
    y_train0 = np.zeros(len(x_train0))
    y_train1 = np.ones(len(x_train1))
    y_train2 = np.repeat(2, len(x_train2))
    y_train3 = np.repeat(3, len(x_train3))
    y_train4 = np.repeat(4, len(x_train4))
    y_train5 = np.repeat(5, len(x_train5))
    y_train6 = np.repeat(6, len(x_train6))
    
    x_train = np.concatenate((x_train0, x_train1, x_train2, x_train3, x_train4, x_train5, x_train6), axis = 0)
    
    y_train = np.concatenate((y_train0, y_train1, y_train2, y_train3, y_train4, y_train5, y_train6), axis = 0)
    
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