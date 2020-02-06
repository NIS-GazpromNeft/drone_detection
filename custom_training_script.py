import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import os

import cv2
import numpy as np

#load all images from a given folder
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

#random brightness and contrast increase/decrease
def add_brightness_contrast(img, 
                            left_brightness_limit = -50, right_brightness_limit = 50,
                            left_contrast_limit = 0, right_contrast_limit = 5):
    
    brightness_increase = np.random.randint(low = left_brightness_limit,
                                            high = right_brightness_limit,
                                            size = 1)
    contrast_increase = np.random.uniform(low = left_contrast_limit,
                                          high = right_contrast_limit,
                                          size = 1)
    
    return cv2.convertScaleAbs(img, alpha = contrast_increase, beta = brightness_increase)

#def focal_loss(y_true, y_predicted, alpha = 4., gamma = 2.):
#    epsilon = 1.e-9
#    y_true = tf.convert_to_tensor(y_true, tf.float32)
#    y_predicted = tf.convert_to_tensor(y_predicted, tf.float32)
#
#    model_out = tf.add(y_predicted, epsilon)
#    ce = tf.multiply(y_true, -tf.log(model_out))
#    weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
#    fl = tf.multiply(alpha, tf.multiply(weight, ce))
#    reduced_fl = tf.reduce_max(fl, axis=1)
    
#    return tf.reduce_mean(reduced_fl)

#one residual block, built from convolution layers followed by batch norm
def residual_block(input_data, n_filters, filter_size):
    
    conv1 = layers.Conv2D(n_filters, filter_size, activation = 'relu', padding = 'same')(input_data)
    bn1 = layers.BatchNormalization()(conv1)
    
    conv2 = layers.Conv2D(n_filters, filter_size, activation = 'relu', padding = 'same')(bn1)
    bn2 = layers.BatchNormalization()(conv2)
    
    out = layers.Add()([bn2, input_data])
    out_activated = layers.Activation('relu')(out)
    
    return out_activated
    
#resnet building
def resnet(x_train, y_train, num_of_classes, val_split = 0.3,
           num_of_residual_blocks = 10, num_of_epochs = 1, 
           dropout_rate = 0.3):
    
    inputs = keras.Input(shape = (256,256,3))
    
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D(3)(x)
    
    for i in range(num_of_residual_blocks):
        x = residual_block(x, 64, 3)
        
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(num_of_classes, activation='softmax')(x)
    
    resnet_model = keras.Model(inputs, outputs)
    
    callbacks = [
            keras.callbacks.TensorBoard(log_dir='./log/custom_training', write_images=True,profile_batch = 100),
            ]
    
    resnet_model.compile(optimizer = keras.optimizers.Adam(),
                          loss = 'sparse_categorical_crossentropy',
                          metrics = ['acc'])
    
    resnet_model.fit(x_train, y_train, epochs = num_of_epochs, validation_split = 0.3, 
                     callbacks = callbacks, batch_size = 32)
    
    #resnet_model.fit(training_data, epochs = num_of_epochs, batch_size = 64,
          #validation_data, validation_steps=3, callbacks=callbacks)

    return resnet_model

#load data
train_path = "save/"
x_train, y_train = data_loader(train_path)

#add random brightness
for index in range(x_train.shape[0]):
    x_train[index] = change_brightness(x_train[index])

num_of_classes = len(np.unique(y_train))    
model = resnet(x_train, y_train, num_of_classes)
model.summary()
model.evaluate(x = x_train, y = y_train)

#test on real data
#test_path = ""
#x_test, y_test = data_loader(test_path)
#model.evaluate(x = x_test, y = y_test)

