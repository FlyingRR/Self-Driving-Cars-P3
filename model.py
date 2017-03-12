import csv
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.layers.core import Dropout
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

samples.pop(0)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


#Flipping
def flip_image(image):
    image_flipped = np.fliplr(image)
    return image_flipped

#Brightness augmentation
def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

#Horizontal and vertical shifts
def trans_image(image,steer,trans_range):
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*0.5
    tr_y = 40*np.random.uniform()-40/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(320,160))
    return image_tr,steer_ang

#Shadow augmentation
def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
 
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image

#Preprocessing
def preprocess_image_file_train(line_data):
    i_lrc = np.random.randint(3)
    y_steer = float(line_data[3])
    if abs(y_steer) > 0.2:
        if (i_lrc == 0):
            #left
            path_file = line_data[1].strip()
            shift_ang = 0.2
        if (i_lrc == 1):
            #center
            path_file = line_data[0].strip()
            shift_ang = 0.
        if (i_lrc == 2):
            #right
            path_file = line_data[2].strip()
            shift_ang = -0.2
    else:
        if (i_lrc == 0):
            #left
            path_file = line_data[1].strip()
            shift_ang = 0.05
        if (i_lrc == 1):
            #center
            path_file = line_data[0].strip()
            shift_ang = 0.
        if (i_lrc == 2):
            #right
            path_file = line_data[2].strip()
            shift_ang = -0.05

    y_steer = float(line_data[3]) + shift_ang
    image = cv2.imread('./data/'+path_file)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image,y_steer = trans_image(image,y_steer,100)
    image = augment_brightness_camera_images(image)
    image = np.array(image)
    ind_flip = np.random.randint(2)
    if ind_flip==0:
        image = cv2.flip(image,1)
        y_steer = -y_steer
    
    return image,y_steer

#Generator
def generate_train_from_PD_batch(data,batch_size = 32):
    num_samples = len(data)
    while 1:
        batch_images = []
        batch_steering = []
        for i_batch in range(batch_size):
            i_line = np.random.randint(num_samples)
            line_data = data[i_line]
            keep_pr = 0
            x,y = preprocess_image_file_train(line_data)

            batch_images.append(x)
            batch_steering.append(y)
        yield np.array(batch_images), np.array(batch_steering)



model = Sequential()

# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/255 - 0.5,
        input_shape=(160,320,3)))

model.add(Convolution2D(24,5,5, activation = 'relu'))
model.add(MaxPooling2D())

model.add(Convolution2D(36,5,5, activation = 'relu'))
model.add(MaxPooling2D())

model.add(Convolution2D(48,5,5, activation = 'relu'))
model.add(MaxPooling2D())

model.add(Convolution2D(64,3,3, activation = 'relu'))
model.add(MaxPooling2D())

model.add(Convolution2D(64,3,3, activation = 'relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam')

# compile and train the model using the generator function
train_generator = generate_train_from_PD_batch(train_samples, batch_size=32)
validation_generator = generate_train_from_PD_batch(validation_samples, batch_size=32)

model.fit_generator(train_generator, 
            samples_per_epoch= 20000, 
            validation_data=validation_generator, 
            nb_val_samples= 400, 
             nb_epoch=8)

model.save('model.h5')
