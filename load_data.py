
import numpy as np

import tensorflow as tf

import keras

training_dir = "./clean_data"
testing_dir = "./test"
batch_size = 400

from keras.preprocessing.image import ImageDataGenerator
from random import randint
from collections import Counter


def data(num_of_images=40,width=105,height=105):

    train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,        zoom_range=0.2,horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(training_dir,        target_size=(width,height),batch_size=batch_size,        class_mode='binary',shuffle=False)

    train = train_generator.next()

    train_x,train_y = train[0],train[1]
    
    if(num_of_images>batch_size):
        exit()
    else: 
        print(Counter(train_output))
        

    train_left_input = np.zeros((num_of_images//2,width,height,3))
    train_right_input = np.zeros((num_of_images//2,width,height,3))
    train_output = np.zeros(num_of_images//2,)
    
    for i in range(num_of_images//2):
        train_left_input[i] = train_x[(((i%40)*10)+randint(0,9))%400]
        
        if randint(0,1):
            train_right_input[i] = train_x[(((i%40)*10)+20)%400]
            train_output[i] = 0
        else:
            train_right_input[i] = train_x[(((i%40)*10)+randint(0,9))%400]
            train_output[i] = 1

    print(Counter(train_output))
    return train_left_input,train_right_input,train_output

from collections import Counter
