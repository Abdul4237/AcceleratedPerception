import math
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom as pydicom
import tensorflow as tf
from custom_layers import MaxPoolingWithArgmax2D, MaxUnpooling2D, conv_block
from PIL import Image
import cv2
from tensorflow import keras
from keras.layers import (Activation, BatchNormalization,
                                     Convolution2D, Input, MaxPool2D,
                                     concatenate)
from tensorflow.keras.models import Model
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

#config = ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.75
#session = InteractiveSession(config=config)


gpu = tf.config.list_physical_devices('GPU')
print(gpu)
for i in gpu:
    tf.config.experimental.set_memory_growth(i, True)
    tf.config.experimental.get_memory_info('GPU:0')



def defineConnSegNet():
    input_images=keras.Input(shape=(256,256,1))

    #Encoder 1:
    enconv_1 = Convolution2D(64,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(input_images)
    enconv_1 = BatchNormalization()(enconv_1)
    enconv_1 = Activation("relu")(enconv_1)
    enconv_1 = Convolution2D(64,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(enconv_1)
    enconv_1 = BatchNormalization()(enconv_1)
    enconv_1 = Activation("relu")(enconv_1)

    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(enconv_1)

    enconv_2 = Convolution2D(128,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(pool_1)
    enconv_2 = BatchNormalization()(enconv_2)
    enconv_2 = Activation("relu")(enconv_2)
    enconv_2 = Convolution2D(128,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(enconv_2)
    enconv_2 = BatchNormalization()(enconv_2)
    enconv_2 = Activation("relu")(enconv_2)


    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(enconv_2)

    enconv_3 = Convolution2D(256,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(pool_2)
    enconv_3 = BatchNormalization()(enconv_3)
    enconv_3 = Activation("relu")(enconv_3)
    enconv_3 = Convolution2D(256,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(enconv_3)
    enconv_3 = BatchNormalization()(enconv_3)
    enconv_3 = Activation("relu")(enconv_3)
    enconv_3 = Convolution2D(256,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(enconv_3)
    enconv_3 = BatchNormalization()(enconv_3)
    enconv_3 = Activation("relu")(enconv_3)

    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(enconv_3)

    enconv_4 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(pool_3)
    enconv_4 = BatchNormalization()(enconv_4)
    enconv_4 = Activation("relu")(enconv_4)
    enconv_4 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(enconv_4)
    enconv_4 = BatchNormalization()(enconv_4)
    enconv_4 = Activation("relu")(enconv_4)
    enconv_4 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(enconv_4)
    enconv_4 = BatchNormalization()(enconv_4)
    enconv_4 = Activation("relu")(enconv_4)
    
    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(enconv_4)

    enconv_5 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(pool_4)
    enconv_5 = BatchNormalization()(enconv_5)
    enconv_5 = Activation("relu")(enconv_5)
    enconv_5 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(enconv_5)
    enconv_5 = BatchNormalization()(enconv_5)
    enconv_5 = Activation("relu")(enconv_5)
    enconv_5 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(enconv_5)
    enconv_5 = BatchNormalization()(enconv_5)
    enconv_5 = Activation("relu")(enconv_5)

    pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(enconv_5)

    #Decoder 1:
    unpool_1 = MaxUnpooling2D()([pool_5, mask_5])

    deconv_6 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(unpool_1)
    deconv_6 = BatchNormalization()(deconv_6)
    deconv_6 = Activation("relu")(deconv_6)
    deconv_6 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(deconv_6)
    deconv_6 = BatchNormalization()(deconv_6)
    deconv_6 = Activation("relu")(deconv_6)
    deconv_6 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(deconv_6)
    deconv_6 = BatchNormalization()(deconv_6)
    deconv_6 = Activation("relu")(deconv_6)

    unpool_2 = MaxUnpooling2D()([deconv_6, mask_4])

    deconv_2 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(unpool_2)
    deconv_2 = BatchNormalization()(deconv_2)
    deconv_2 = Activation("relu")(deconv_2)
    deconv_2 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(deconv_2)
    deconv_2 = BatchNormalization()(deconv_2)
    deconv_2 = Activation("relu")(deconv_2)

    deconv_2 = Convolution2D(256,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(deconv_2)
    deconv_2 = BatchNormalization()(deconv_2)
    deconv_2 = Activation("relu")(deconv_2)
    
    unpool_3 = MaxUnpooling2D()([deconv_2, mask_3])

    deconv_3 = Convolution2D(256,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(unpool_3)
    deconv_3 = BatchNormalization()(deconv_3)
    deconv_3 = Activation("relu")(deconv_3)
    deconv_3 = Convolution2D(256,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(deconv_3)
    deconv_3 = BatchNormalization()(deconv_3)
    deconv_3 = Activation("relu")(deconv_3)

    deconv_3 = Convolution2D(128,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(deconv_3)
    deconv_3 = BatchNormalization()(deconv_3)
    deconv_3= Activation("relu")(deconv_3)

    unpool_4 = MaxUnpooling2D()([deconv_3, mask_2])

    deconv_4 = Convolution2D(128,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(unpool_4)
    deconv_4 = BatchNormalization()(deconv_4)
    deconv_4= Activation("relu")(deconv_4)
    deconv_4 = Convolution2D(128,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(deconv_4)
    deconv_4 = BatchNormalization()(deconv_4)
    deconv_4 = Activation("relu")(deconv_4)

    deconv_4= Convolution2D(64,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(deconv_4)
    deconv_4 = BatchNormalization()(deconv_4)
    deconv_4= Activation("relu")(deconv_4)

    unpool_5 = MaxUnpooling2D()([deconv_4, mask_1])

    #Encode 2:
    conv_13 = Convolution2D(64,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(unpool_5)
    conv_13 = BatchNormalization()(conv_13)
    conv_13= Activation("relu")(conv_13)
    conv_13 = Convolution2D(64,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(conv_13)
    conv_13 = BatchNormalization()(conv_13)
    conv_13 = Activation("relu")(conv_13)
    conv_13 = Convolution2D(64,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(conv_13)
    conv_13 = BatchNormalization()(conv_13)
    conv_13 = Activation("relu")(conv_13)
    
    pool_6, mask_6 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv_13)
    merge_1=concatenate([pool_6,deconv_4],axis=3)

    enconv_14 = Convolution2D(128,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(merge_1)
    enconv_14 = BatchNormalization()(enconv_14)
    enconv_14= Activation("relu")(enconv_14)
    enconv_14 = Convolution2D(128,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(enconv_14)
    enconv_14 = BatchNormalization()(enconv_14)
    enconv_14= Activation("relu")(enconv_14)

    pool_7, mask_7 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(enconv_14)
    merge_2=concatenate([pool_7,deconv_3],axis=3)

    enconv_15 = Convolution2D(256,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(merge_2)
    enconv_15 = BatchNormalization()(enconv_15)
    enconv_15= Activation("relu")(enconv_15)
    enconv_15 = Convolution2D(256,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(enconv_15)
    enconv_15 = BatchNormalization()(enconv_15)
    enconv_15= Activation("relu")(enconv_15)
    enconv_15 = Convolution2D(256,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(enconv_15)
    enconv_15 = BatchNormalization()(enconv_15)
    enconv_15= Activation("relu")(enconv_15)

    pool_8, mask_8 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(enconv_15)
    merge_3=concatenate([pool_8,deconv_2],axis=3)

    enconv_16 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(merge_3)
    enconv_16 = BatchNormalization()(enconv_16)
    enconv_16= Activation("relu")(enconv_16)
    enconv_16 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(enconv_16)
    enconv_16 = BatchNormalization()(enconv_16)
    enconv_16= Activation("relu")(enconv_16)
    enconv_16 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(enconv_16)
    enconv_16 = BatchNormalization()(enconv_16)
    enconv_16= Activation("relu")(enconv_16)

    pool_9, mask_9 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(enconv_16)
    merge_4=concatenate([pool_9,deconv_6],axis=3)

    enconv_17 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(merge_4)
    enconv_17 = BatchNormalization()(enconv_17)
    enconv_17= Activation("relu")(enconv_17)
    enconv_17 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(enconv_17)
    enconv_17 = BatchNormalization()(enconv_17)
    enconv_17= Activation("relu")(enconv_17)
    enconv_17 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(enconv_17)
    enconv_17 = BatchNormalization()(enconv_17)
    enconv_17= Activation("relu")(enconv_17)

    pool_10, mask_10 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(enconv_17)
    #Decode 2:
    unpool_6 = MaxUnpooling2D()([pool_10, mask_10])

    enconv_18 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(unpool_6)
    enconv_18 = BatchNormalization()(enconv_18)
    enconv_18= Activation("relu")(enconv_18)
    enconv_18 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(enconv_18)
    enconv_18 = BatchNormalization()(enconv_18)
    enconv_18= Activation("relu")(enconv_18)
    enconv_18 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(enconv_18)
    enconv_18 = BatchNormalization()(enconv_18)
    enconv_18= Activation("relu")(enconv_18)

    unpool_7 = MaxUnpooling2D()([enconv_18, mask_9])

    enconv_19= Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(unpool_7)
    enconv_19 = BatchNormalization()(enconv_19)
    enconv_19= Activation("relu")(enconv_19)
    enconv_19= Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(enconv_19)
    enconv_19 = BatchNormalization()(enconv_19)
    enconv_19= Activation("relu")(enconv_19)
    enconv_20= Convolution2D(256,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(enconv_19)
    enconv_20 = BatchNormalization()(enconv_20)
    enconv_20= Activation("relu")(enconv_20)

    unpool_8 = MaxUnpooling2D()([enconv_20, mask_8])

    enconv_21= Convolution2D(256,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(unpool_8)
    enconv_21 = BatchNormalization()(enconv_21)
    enconv_21= Activation("relu")(enconv_21)
    enconv_21= Convolution2D(256,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(enconv_21)
    enconv_21 = BatchNormalization()(enconv_21)
    enconv_21= Activation("relu")(enconv_21)
    enconv_22= Convolution2D(128,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(enconv_21)
    enconv_22 = BatchNormalization()(enconv_22)
    enconv_22= Activation("relu")(enconv_22)

    unpool_9 = MaxUnpooling2D()([enconv_22, mask_7])
    
    enconv_23= Convolution2D(128,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(unpool_9)
    enconv_23 = BatchNormalization()(enconv_23)
    enconv_23= Activation("relu")(enconv_23)
    enconv_23= Convolution2D(128,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(enconv_23)
    enconv_23 = BatchNormalization()(enconv_23)
    enconv_23= Activation("relu")(enconv_23)
    enconv_24= Convolution2D(64,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(enconv_23)
    enconv_24 = BatchNormalization()(enconv_24)
    enconv_24= Activation("relu")(enconv_24)

    unpool_10 = MaxUnpooling2D()([enconv_24, mask_6])

    enconv_25= Convolution2D(64,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(unpool_10)
    enconv_25 = BatchNormalization()(enconv_25)
    enconv_25= Activation("relu")(enconv_25)
    deconv_26= Convolution2D(64,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform',dilation_rate=(3,3))(enconv_25)
    deconv_26 = BatchNormalization()(deconv_26)
    deconv_26= Activation("relu")(deconv_26)
    deconv_27= Convolution2D(1,1,strides=(1, 1), padding="same", kernel_initializer='random_uniform',)(deconv_26)
    output= Activation("relu")(deconv_27)
    output=tf.clip_by_value(output,clip_value_min=0,clip_value_max=1)
    model = keras.Model(inputs=input_images, outputs=output, name="Connected-SegNets")
    return model

def defineSegNet():
    input_images=keras.Input(shape=(256,256,1))
    enconv_1 = Convolution2D(64,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(input_images)
    enconv_1 = BatchNormalization()(enconv_1)
    enconv_1 = Activation("relu")(enconv_1)
    enconv_1 = Convolution2D(64,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(enconv_1)
    enconv_1 = BatchNormalization()(enconv_1)
    enconv_1 = Activation("relu")(enconv_1) 

    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(enconv_1)

    enconv_2 = Convolution2D(128,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(pool_1)
    enconv_2 = BatchNormalization()(enconv_2)
    enconv_2 = Activation("relu")(enconv_2)
    enconv_2 = Convolution2D(128,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(enconv_2)
    enconv_2 = BatchNormalization()(enconv_2)
    enconv_2 = Activation("relu")(enconv_2)


    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(enconv_2)

    enconv_3 = Convolution2D(256,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(pool_2)
    enconv_3 = BatchNormalization()(enconv_3)
    enconv_3 = Activation("relu")(enconv_3)
    enconv_3 = Convolution2D(256,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(enconv_3)
    enconv_3 = BatchNormalization()(enconv_3)
    enconv_3 = Activation("relu")(enconv_3)
    enconv_3 = Convolution2D(256,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(enconv_3)
    enconv_3 = BatchNormalization()(enconv_3)
    enconv_3 = Activation("relu")(enconv_3)

    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(enconv_3)

    enconv_4 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(pool_3)
    enconv_4 = BatchNormalization()(enconv_4)
    enconv_4 = Activation("relu")(enconv_4)
    enconv_4 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(enconv_4)
    enconv_4 = BatchNormalization()(enconv_4)
    enconv_4 = Activation("relu")(enconv_4)
    enconv_4 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(enconv_4)
    enconv_4 = BatchNormalization()(enconv_4)
    enconv_4 = Activation("relu")(enconv_4)

    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(enconv_4)

    enconv_5 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(pool_4)
    enconv_5 = BatchNormalization()(enconv_5)
    enconv_5 = Activation("relu")(enconv_5)
    enconv_5 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(enconv_5)
    enconv_5 = BatchNormalization()(enconv_5)
    enconv_5 = Activation("relu")(enconv_5)
    enconv_5 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(enconv_5)
    enconv_5 = BatchNormalization()(enconv_5)
    enconv_5 = Activation("relu")(enconv_5)

    pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(enconv_5)
   
   #Decoder 1:
    unpool_1 = MaxUnpooling2D()([pool_5, mask_5])

    deconv_1 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(unpool_1)
    deconv_1 = BatchNormalization()(deconv_1)
    deconv_1 = Activation("relu")(deconv_1)
    deconv_1 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(deconv_1)
    deconv_1 = BatchNormalization()(deconv_1)
    deconv_1 = Activation("relu")(deconv_1)
    deconv_1 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(deconv_1)
    deconv_1 = BatchNormalization()(deconv_1)
    deconv_1 = Activation("relu")(deconv_1)

    unpool_2 = MaxUnpooling2D()([deconv_1, mask_4])

    deconv_2 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(unpool_2)
    deconv_2 = BatchNormalization()(deconv_2)
    deconv_2 = Activation("relu")(deconv_2)
    deconv_2 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(deconv_2)
    deconv_2 = BatchNormalization()(deconv_2)
    deconv_2 = Activation("relu")(deconv_2)

    deconv_2 = Convolution2D(256,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(deconv_2)
    deconv_2 = BatchNormalization()(deconv_2)
    deconv_2 = Activation("relu")(deconv_2)
    
    unpool_3 = MaxUnpooling2D()([deconv_2, mask_3])

    deconv_3 = Convolution2D(256,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(unpool_3)
    deconv_3 = BatchNormalization()(deconv_3)
    deconv_3 = Activation("relu")(deconv_3)
    deconv_3 = Convolution2D(256,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(deconv_3)
    deconv_3 = BatchNormalization()(deconv_3)
    deconv_3 = Activation("relu")(deconv_3)

    deconv_3 = Convolution2D(128,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(deconv_3)
    deconv_3 = BatchNormalization()(deconv_3)
    deconv_3= Activation("relu")(deconv_3)

    unpool_4 = MaxUnpooling2D()([deconv_3, mask_2])

    deconv_4 = Convolution2D(128,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(unpool_4)
    deconv_4 = BatchNormalization()(deconv_4)
    deconv_4= Activation("relu")(deconv_4)
    deconv_4 = Convolution2D(64,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(deconv_4)
    deconv_4 = BatchNormalization()(deconv_4)
    deconv_4 = Activation("relu")(deconv_4)

    unpool_5 = MaxUnpooling2D()([deconv_4, mask_1])

    deconv_5= Convolution2D(64,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform')(unpool_5)
    deconv_5 = BatchNormalization()(deconv_5)
    deconv_5= Activation("relu")(deconv_5)
    #try dilation_rate=(3,3)
    deconv_5= Convolution2D(64,3,strides=(1, 1), padding="same", kernel_initializer='random_uniform',dilation_rate=(3,3))(deconv_5)
    deconv_5 = BatchNormalization()(deconv_5)
    deconv_5= Activation("relu")(deconv_5)
    deconv_5= Convolution2D(1,1,strides=(1, 1), padding="same", kernel_initializer='random_uniform',)(deconv_5)

    #change number of filters
    output=tf.keras.activations.sigmoid(deconv_5)
    #output= Activation("relu")(deconv_5)
    #output=tf.clip_by_value(output,clip_value_min=0,clip_value_max=1)
    model = keras.Model(inputs=input_images, outputs=output, name="SegNets")
    return model
  
def unet3plus(input_shape, output_channels):
    """ UNet3+ base model """
    filters = [64, 128, 256, 512, 1024]

    input_layer = keras.layers.Input(
        shape=input_shape,
        name="input_layer"
    )  # 320*320*3

    """ Encoder"""
    # block 1
    e1 = conv_block(input_layer, filters[0])  # 320*320*64

    # block 2
    e2 = keras.layers.MaxPool2D(pool_size=(2, 2))(e1)  # 160*160*64
    e2 = conv_block(e2, filters[1])  # 160*160*128

    # block 3
    e3 = keras.layers.MaxPool2D(pool_size=(2, 2))(e2)  # 80*80*128
    e3 = conv_block(e3, filters[2])  # 80*80*256

    # block 4
    e4 = keras.layers.MaxPool2D(pool_size=(2, 2))(e3)  # 40*40*256
    e4 = conv_block(e4, filters[3])  # 40*40*512

    # block 5
    # bottleneck layer
    e5 = keras.layers.MaxPool2D(pool_size=(2, 2))(e4)  # 20*20*512
    e5 = conv_block(e5, filters[4])  # 20*20*1024

    """ Decoder """
    cat_channels = filters[0]
    cat_blocks = len(filters)
    upsample_channels = cat_blocks * cat_channels

    """ d4 """
    e1_d4 = keras.layers.MaxPool2D(pool_size=(8, 8))(e1)  # 320*320*64  --> 40*40*64
    e1_d4 = conv_block(e1_d4, cat_channels, n=1)  # 320*320*64  --> 40*40*64

    e2_d4 = keras.layers.MaxPool2D(pool_size=(4, 4))(e2)  # 160*160*128 --> 40*40*128
    e2_d4 = conv_block(e2_d4, cat_channels, n=1)  # 160*160*128 --> 40*40*64

    e3_d4 = keras.layers.MaxPool2D(pool_size=(2, 2))(e3)  # 80*80*256  --> 40*40*256
    e3_d4 = conv_block(e3_d4, cat_channels, n=1)  # 80*80*256  --> 40*40*64

    e4_d4 = conv_block(e4, cat_channels, n=1)  # 40*40*512  --> 40*40*64

    e5_d4 = keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(e5)  # 80*80*256  --> 40*40*256
    e5_d4 = conv_block(e5_d4, cat_channels, n=1)  # 20*20*1024  --> 20*20*64

    d4 = keras.layers.concatenate([e1_d4, e2_d4, e3_d4, e4_d4, e5_d4])
    d4 = conv_block(d4, upsample_channels, n=1)  # 40*40*320  --> 40*40*320

    """ d3 """
    e1_d3 = keras.layers.MaxPool2D(pool_size=(4, 4))(e1)  # 320*320*64 --> 80*80*64
    e1_d3 = conv_block(e1_d3, cat_channels, n=1)  # 80*80*64 --> 80*80*64

    e2_d3 = keras.layers.MaxPool2D(pool_size=(2, 2))(e2)  # 160*160*256 --> 80*80*256
    e2_d3 = conv_block(e2_d3, cat_channels, n=1)  # 80*80*256 --> 80*80*64

    e3_d3 = conv_block(e3, cat_channels, n=1)  # 80*80*512 --> 80*80*64

    e4_d3 = keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(d4)  # 40*40*320 --> 80*80*320
    e4_d3 = conv_block(e4_d3, cat_channels, n=1)  # 80*80*320 --> 80*80*64

    e5_d3 = keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(e5)  # 20*20*320 --> 80*80*320
    e5_d3 = conv_block(e5_d3, cat_channels, n=1)  # 80*80*320 --> 80*80*64

    d3 = keras.layers.concatenate([e1_d3, e2_d3, e3_d3, e4_d3, e5_d3])
    d3 = conv_block(d3, upsample_channels, n=1)  # 80*80*320 --> 80*80*320

    """ d2 """
    e1_d2 = keras.layers.MaxPool2D(pool_size=(2, 2))(e1)  # 320*320*64 --> 160*160*64
    e1_d2 = conv_block(e1_d2, cat_channels, n=1)  # 160*160*64 --> 160*160*64

    e2_d2 = conv_block(e2, cat_channels, n=1)  # 160*160*256 --> 160*160*64

    d3_d2 = keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(d3)  # 80*80*320 --> 160*160*320
    d3_d2 = conv_block(d3_d2, cat_channels, n=1)  # 160*160*320 --> 160*160*64

    d4_d2 = keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(d4)  # 40*40*320 --> 160*160*320
    d4_d2 = conv_block(d4_d2, cat_channels, n=1)  # 160*160*320 --> 160*160*64

    e5_d2 = keras.layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(e5)  # 20*20*320 --> 160*160*320
    e5_d2 = conv_block(e5_d2, cat_channels, n=1)  # 160*160*320 --> 160*160*64

    d2 = keras.layers.concatenate([e1_d2, e2_d2, d3_d2, d4_d2, e5_d2])
    d2 = conv_block(d2, upsample_channels, n=1)  # 160*160*320 --> 160*160*320

    """ d1 """
    e1_d1 = conv_block(e1, cat_channels, n=1)  # 320*320*64 --> 320*320*64

    d2_d1 = keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(d2)  # 160*160*320 --> 320*320*320
    d2_d1 = conv_block(d2_d1, cat_channels, n=1)  # 160*160*320 --> 160*160*64

    d3_d1 = keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(d3)  # 80*80*320 --> 320*320*320
    d3_d1 = conv_block(d3_d1, cat_channels, n=1)  # 320*320*320 --> 320*320*64

    d4_d1 = keras.layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(d4)  # 40*40*320 --> 320*320*320
    d4_d1 = conv_block(d4_d1, cat_channels, n=1)  # 320*320*320 --> 320*320*64

    e5_d1 = keras.layers.UpSampling2D(size=(16, 16), interpolation='bilinear')(e5)  # 20*20*320 --> 320*320*320
    e5_d1 = conv_block(e5_d1, cat_channels, n=1)  # 320*320*320 --> 320*320*64

    d1 = keras.layers.concatenate([e1_d1, d2_d1, d3_d1, d4_d1, e5_d1, ])
    d1 = conv_block(d1, upsample_channels, n=1)  # 320*320*320 --> 320*320*320

    # last layer does not have batchnorm and relu
    d = conv_block(d1, output_channels, n=1, is_bn=False, is_relu=False)

    output = keras.activations.softmax(d)

    return tf.keras.Model(inputs=input_layer, outputs=[output], name='UNet_3Plus')

model=unet3plus((256,256,1),1)
#model.summary()
model.compile(
    loss=keras.losses.BinaryCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    metrics=['accuracy'],
)

def get_pixels(dcm_file):
    im = pydicom.dcmread(dcm_file)
    
    data = im.pixel_array
    
    if im.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    else:
        data = data - np.min(data)
        
    if np.max(data) != 0:
        data = data / np.max(data)
    data=(data * 255).astype(np.uint8)

    return data
def get_pixels_with_windowing(dcm_file):
    im = pydicom.dcmread(dcm_file)
    
    data = im.pixel_array
    
    # This line is the only difference in the two functions
    data = pydicom.pixel_data_handlers.apply_windowing(data, im)
    
    if im.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    else:
        data = data - np.min(data)
        
    if np.max(data) != 0:
        data = data / np.max(data)
    data=(data * 255).astype(np.uint8)

    return data
def resize_image(image):
    # scale the image
    image = tf.cast(image, tf.float32)
    image = image/255.0
    # resize image
    image = tf.image.resize(image, (256,256))
    return image
def resize_mask(mask):
    # resize the mask
    mask = tf.image.resize(mask, (256,256))
    mask = tf.cast(mask, tf.uint8)
    return mask   
def findMaskBox(mask):
  left,right,top,bottom=0,0,0,0
  foundLeft,foundTop=False,False
  #finding bounding box
  for i in range(mask.shape[1]):
    if np.max(mask[:,[i]]) >0 and foundLeft==False:
      left=i-1
      foundLeft=True
    elif np.max(mask[:,[i]]) ==0 and foundLeft==True and i>left+30 or i==mask.shape[1]-1:
      right=i
      break

  for s in range(mask.shape[0]):
    if np.max(mask[[s],:]) >0 and foundTop==False:
      top=s-1
      foundTop=True
    elif np.max(mask[[s],:]) ==0 and foundTop==True and s>top+30 or s==mask.shape[0]-1:
      bottom=s
      break
  height=bottom-top
  width=right-left
  #print(mask.shape)
  #print(left,right,top,bottom)
  #print(height,width)
  #creating bounding box square by adding extra pixels to smaller length
  if height>width:
    dif=height-width
    update=math.floor(dif/2)
    left=left - update
    if dif % 2 != 0:
      right=right + update + 1
    else:
      right=right +update
    width=right-left
  elif width>height:
    dif=width-height
    update=math.floor(dif/2)
    top=top-update
    if dif % 2 != 0:
      bottom=bottom + update + 1
    else:
      bottom=bottom +update
    height=bottom-top
  #print(height,width)
  return left,right,top,bottom
def findFiles(path):
    paths=[]
    for subdir,dir, files in os.walk(path):
        for file in files:
            paths.append(os.path.join(subdir,file))
    #print(paths)
    return paths
#data = pd.read_csv(r"E:\ADMANI\manifest-ZkhPvrLo5216730872708713142\mass_case_description_train_set.csv")
#base=r'e:\ADMANI\manifest-ZkhPvrLo5216730872708713142\CBIS-DDSM'
#finding all images for every patient
numimages=1318
lastid=0
lastview=''
lastlat=''
numsSeen=[]
#data preprocessing and extraction
for x in range(numimages):
  break
  #fetch values from csv
  laterality=data._get_value(x,'left or right breast')
  view=data._get_value(x,'image view')
  id=data._get_value(x,'patient_id') 
  number=data._get_value(x,'abnormality id')
  pathList=[]
  mamImages=[]

  if id!=lastid:
    numsSeen=[]
    lastid=id
    print(lastid+"_"+str(laterality)+str(view)+"_"+str(number))
    path=f"Mass-Training_{id}_{laterality}_{view}"
    path=os.path.join(base,path)
    mamPath=findFiles(path)[0]
    #finding 1st mask
    path2=path+'_1'
    imgs=findFiles(path2)
    if os.path.getsize(imgs[0])>os.path.getsize(imgs[1]):
        maskPath=imgs[0]
    else:
      maskPath=imgs[1]
    pathList=[mamPath,maskPath]
    numsSeen.append(1)


  #add additional masks from same patient(if)
  elif id == lastid and number not in numsSeen:
    print(lastid+"_"+str(laterality)+str(view)+"_"+str(number))
    path=f"Mass-Training_{id}_{laterality}_{view}"
    path=os.path.join(base,path)
    mamPath=findFiles(path)[0]
    path2= path + f'_{number}'
    imgs=findFiles(path2)
    if os.path.getsize(imgs[0])>os.path.getsize(imgs[1]):
      maskPath=imgs[0]
    else:
      maskPath=imgs[1]
    pathList=[mamPath,maskPath]
    numsSeen.append(number)

  elif id == lastid and number==1:
    continue
  
  
  for i in pathList:
    image = get_pixels_with_windowing(i)
    #adding images into lists
    if pathList.index(i)==0:
      mamImages.append(image)
    else:
    # finding pixel cordinates of corners
      mask=image
      #plt.imshow(mask,cmap='gray')
      #plt.show()
      left,right,top,bottom=findMaskBox(mask)
    #cropping out mask
      mask = Image.fromarray(mask.astype('uint8'), 'L')
      mask=mask.crop((left,top,right,bottom))
    #cropping out ROI from mammogram
      roi=mamImages[0]
      #plt.imshow(roi,cmap='gray')
      #plt.show()
      roi = Image.fromarray(roi.astype('uint8'), 'L')
      roi=roi.crop((left,top,right,bottom))
      #checking if mask is valid
      if np.max(np.array(mask))==255:
        pass
      else:
         print(f'False:{x}')
    #displaying images
      #fig = plt.figure(figsize=(10, 7))
      #fig.add_subplot(2,2, 1)
      #plt.imshow(mask,cmap='gray')
      #fig.add_subplot(2,2,2)
      #plt.imshow(roi,cmap='gray')
      #plt.show()
    #applying CLAHE on ROI's
      clahe = cv2.createCLAHE(clipLimit=1)
      final_img = clahe.apply(np.array(roi)) + 10
      #fig.add_subplot(2,2,2)
      #plt.imshow(final_img,cmap='gray')
      #fig.add_subplot(2,2,1)
      #plt.imshow(roi,cmap='gray')
      #plt.show()
    #making image model ready by expanding dims(removed)
      clahe = Image.fromarray(final_img.astype('uint8'), 'L')
      clahe.save(f'E:\ADMANI\ROI4clahe\{str(id)+"_"+str(laterality)+"_"+str(view)+"_"+str(number)}.bmp')
      roi.save(f'E:\ADMANI\ROI4\{str(id)+"_"+str(laterality)+"_"+str(view)+"_"+str(number)}.bmp')
      mask.save(f'E:\ADMANI\ROI4masks\{str(id)+"_"+str(laterality)+"_"+str(view)+"_"+str(number)}.bmp')

data_gen_args = dict(featurewise_center=False,
                    validation_split=0.15,
                    rescale=1./255,
                    featurewise_std_normalization=False,
                    rotation_range=180,
                    horizontal_flip=True)
image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
seed=2
image_generator = image_datagen.flow_from_directory(
    r'E:\ADMANI\ROI4clahe',
    class_mode=None,
    target_size=(256,256),
    color_mode='grayscale',
    batch_size=2,
    keep_aspect_ratio=True,
    seed=seed)
mask_generator = mask_datagen.flow_from_directory(
    r'E:\ADMANI\ROI4masks',
    target_size=(256,256),
    color_mode='grayscale',
    class_mode=None,
    batch_size=2,
    keep_aspect_ratio=True,
    seed=seed)

# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)

history=model.fit(
    train_generator,
    epochs=10,
    #validation_data=train_generator,
    #validation_steps=25,
    steps_per_epoch=186*8
    #callbacks=[cp_callback]
    )
