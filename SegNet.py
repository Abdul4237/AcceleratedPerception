import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPool2D
from custom_layers import MaxPoolingWithArgmax2D
from custom_layers import MaxUnpooling2D
import numpy as np
from PIL import Image
import os
import pandas as pd
import cv2
import pydicom as dicom
import matplotlib.pyplot as plt
gpu = tf.config.list_physical_devices('GPU')
for i in gpu:
    tf.config.experimental.set_memory_growth(i, True)

def defineNetwork():
    input_images=keras.Input(shape=(256,256,1))

    #Encoder 1:
    enconv_1 = Convolution2D(64,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(input_images)
    enconv_1 = BatchNormalization()(enconv_1)
    enconv_1 = Activation("relu")(enconv_1)
    enconv_1 = Convolution2D(64,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(enconv_1)
    enconv_1 = BatchNormalization()(enconv_1)
    enconv_1 = Activation("relu")(enconv_1)

    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(enconv_1)

    enconv_2 = Convolution2D(128,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(pool_1)
    enconv_2 = BatchNormalization()(enconv_2)
    enconv_2 = Activation("relu")(enconv_2)
    enconv_2 = Convolution2D(128,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(enconv_2)
    enconv_2 = BatchNormalization()(enconv_2)
    enconv_2 = Activation("relu")(enconv_2)


    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(enconv_2)

    enconv_3 = Convolution2D(256,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(pool_2)
    enconv_3 = BatchNormalization()(enconv_3)
    enconv_3 = Activation("relu")(enconv_3)
    enconv_3 = Convolution2D(256,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(enconv_3)
    enconv_3 = BatchNormalization()(enconv_3)
    enconv_3 = Activation("relu")(enconv_3)
    enconv_3 = Convolution2D(256,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(enconv_3)
    enconv_3 = BatchNormalization()(enconv_3)
    enconv_3 = Activation("relu")(enconv_3)

    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(enconv_3)

    enconv_4 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(pool_3)
    enconv_4 = BatchNormalization()(enconv_4)
    enconv_4 = Activation("relu")(enconv_4)
    enconv_4 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(enconv_4)
    enconv_4 = BatchNormalization()(enconv_4)
    enconv_4 = Activation("relu")(enconv_4)
    enconv_4 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(enconv_4)
    enconv_4 = BatchNormalization()(enconv_4)
    enconv_4 = Activation("relu")(enconv_4)
    
    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(enconv_4)

    enconv_5 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(pool_4)
    enconv_5 = BatchNormalization()(enconv_5)
    enconv_5 = Activation("relu")(enconv_5)
    enconv_5 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(enconv_5)
    enconv_5 = BatchNormalization()(enconv_5)
    enconv_5 = Activation("relu")(enconv_5)
    enconv_5 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(enconv_5)
    enconv_5 = BatchNormalization()(enconv_5)
    enconv_5 = Activation("relu")(enconv_5)

    pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(enconv_5)

    #Decoder 1:
    unpool_1 = MaxUnpooling2D()([pool_5, mask_5])

    deconv_6 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(unpool_1)
    deconv_6 = BatchNormalization()(deconv_6)
    deconv_6 = Activation("relu")(deconv_6)
    deconv_6 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(deconv_6)
    deconv_6 = BatchNormalization()(deconv_6)
    deconv_6 = Activation("relu")(deconv_6)
    deconv_6 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(deconv_6)
    deconv_6 = BatchNormalization()(deconv_6)
    deconv_6 = Activation("relu")(deconv_6)

    unpool_2 = MaxUnpooling2D()([deconv_6, mask_4])

    deconv_7 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(unpool_2)
    deconv_7 = BatchNormalization()(deconv_7)
    deconv_7 = Activation("relu")(deconv_7)
    deconv_7 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(deconv_7)
    deconv_7 = BatchNormalization()(deconv_7)
    deconv_7 = Activation("relu")(deconv_7)

    deconv_8 = Convolution2D(256,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(deconv_7)
    deconv_8 = BatchNormalization()(deconv_8)
    deconv_8 = Activation("relu")(deconv_8)
    
    unpool_3 = MaxUnpooling2D()([deconv_8, mask_3])

    deconv_9 = Convolution2D(256,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(unpool_3)
    deconv_9 = BatchNormalization()(deconv_9)
    deconv_9 = Activation("relu")(deconv_9)
    deconv_9 = Convolution2D(256,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(deconv_9)
    deconv_9 = BatchNormalization()(deconv_9)
    deconv_9 = Activation("relu")(deconv_9)

    deconv_10 = Convolution2D(128,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(deconv_9)
    deconv_10 = BatchNormalization()(deconv_10)
    deconv_10= Activation("relu")(deconv_10)

    unpool_4 = MaxUnpooling2D()([deconv_10, mask_2])

    deconv_11 = Convolution2D(128,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(unpool_4)
    deconv_11 = BatchNormalization()(deconv_11)
    deconv_11= Activation("relu")(deconv_11)
    deconv_11 = Convolution2D(128,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(deconv_11)
    deconv_11 = BatchNormalization()(deconv_11)
    deconv_11 = Activation("relu")(deconv_11)

    deconv_12= Convolution2D(64,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(deconv_11)
    deconv_12 = BatchNormalization()(deconv_12)
    deconv_12= Activation("relu")(deconv_12)

    unpool_5 = MaxUnpooling2D()([deconv_12, mask_1])

    #Encode 2:
    conv_13 = Convolution2D(64,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(unpool_5)
    conv_13 = BatchNormalization()(conv_13)
    conv_13= Activation("relu")(conv_13)
    conv_13 = Convolution2D(64,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(conv_13)
    conv_13 = BatchNormalization()(conv_13)
    conv_13 = Activation("relu")(conv_13)
    conv_13 = Convolution2D(64,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(conv_13)
    conv_13 = BatchNormalization()(conv_13)
    conv_13 = Activation("relu")(conv_13)
    
    pool_6, mask_6 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv_13)
    merge_1=concatenate([pool_6,deconv_12],axis=3)

    enconv_14 = Convolution2D(128,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(merge_1)
    enconv_14 = BatchNormalization()(enconv_14)
    enconv_14= Activation("relu")(enconv_14)
    enconv_14 = Convolution2D(128,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(enconv_14)
    enconv_14 = BatchNormalization()(enconv_14)
    enconv_14= Activation("relu")(enconv_14)

    pool_7, mask_7 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(enconv_14)
    merge_2=concatenate([pool_7,deconv_10],axis=3)

    enconv_15 = Convolution2D(256,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(merge_2)
    enconv_15 = BatchNormalization()(enconv_15)
    enconv_15= Activation("relu")(enconv_15)
    enconv_15 = Convolution2D(256,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(enconv_15)
    enconv_15 = BatchNormalization()(enconv_15)
    enconv_15= Activation("relu")(enconv_15)
    enconv_15 = Convolution2D(256,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(enconv_15)
    enconv_15 = BatchNormalization()(enconv_15)
    enconv_15= Activation("relu")(enconv_15)

    pool_8, mask_8 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(enconv_15)
    merge_3=concatenate([pool_8,deconv_8],axis=3)

    enconv_16 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(merge_3)
    enconv_16 = BatchNormalization()(enconv_16)
    enconv_16= Activation("relu")(enconv_16)
    enconv_16 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(enconv_16)
    enconv_16 = BatchNormalization()(enconv_16)
    enconv_16= Activation("relu")(enconv_16)
    enconv_16 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(enconv_16)
    enconv_16 = BatchNormalization()(enconv_16)
    enconv_16= Activation("relu")(enconv_16)

    pool_9, mask_9 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(enconv_16)
    merge_4=concatenate([pool_9,deconv_6],axis=3)

    enconv_17 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(merge_4)
    enconv_17 = BatchNormalization()(enconv_17)
    enconv_17= Activation("relu")(enconv_17)
    enconv_17 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(enconv_17)
    enconv_17 = BatchNormalization()(enconv_17)
    enconv_17= Activation("relu")(enconv_17)
    enconv_17 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(enconv_17)
    enconv_17 = BatchNormalization()(enconv_17)
    enconv_17= Activation("relu")(enconv_17)

    pool_10, mask_10 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(enconv_17)
    #Decode 2:
    unpool_6 = MaxUnpooling2D()([pool_10, mask_10])

    enconv_18 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(unpool_6)
    enconv_18 = BatchNormalization()(enconv_18)
    enconv_18= Activation("relu")(enconv_18)
    enconv_18 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(enconv_18)
    enconv_18 = BatchNormalization()(enconv_18)
    enconv_18= Activation("relu")(enconv_18)
    enconv_18 = Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(enconv_18)
    enconv_18 = BatchNormalization()(enconv_18)
    enconv_18= Activation("relu")(enconv_18)

    unpool_7 = MaxUnpooling2D()([enconv_18, mask_9])

    enconv_19= Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(unpool_7)
    enconv_19 = BatchNormalization()(enconv_19)
    enconv_19= Activation("relu")(enconv_19)
    enconv_19= Convolution2D(512,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(enconv_19)
    enconv_19 = BatchNormalization()(enconv_19)
    enconv_19= Activation("relu")(enconv_19)
    enconv_20= Convolution2D(256,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(enconv_19)
    enconv_20 = BatchNormalization()(enconv_20)
    enconv_20= Activation("relu")(enconv_20)

    unpool_8 = MaxUnpooling2D()([enconv_20, mask_8])

    enconv_21= Convolution2D(256,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(unpool_8)
    enconv_21 = BatchNormalization()(enconv_21)
    enconv_21= Activation("relu")(enconv_21)
    enconv_21= Convolution2D(256,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(enconv_21)
    enconv_21 = BatchNormalization()(enconv_21)
    enconv_21= Activation("relu")(enconv_21)
    enconv_22= Convolution2D(128,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(enconv_21)
    enconv_22 = BatchNormalization()(enconv_22)
    enconv_22= Activation("relu")(enconv_22)

    unpool_9 = MaxUnpooling2D()([enconv_22, mask_7])
    
    enconv_23= Convolution2D(128,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(unpool_9)
    enconv_23 = BatchNormalization()(enconv_23)
    enconv_23= Activation("relu")(enconv_23)
    enconv_23= Convolution2D(128,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(enconv_23)
    enconv_23 = BatchNormalization()(enconv_23)
    enconv_23= Activation("relu")(enconv_23)
    enconv_24= Convolution2D(64,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(enconv_23)
    enconv_24 = BatchNormalization()(enconv_24)
    enconv_24= Activation("relu")(enconv_24)

    unpool_10 = MaxUnpooling2D()([enconv_24, mask_6])

    enconv_25= Convolution2D(64,3,strides=(1, 1), padding="same", kernel_initializer='he_normal')(unpool_10)
    enconv_25 = BatchNormalization()(enconv_25)
    enconv_25= Activation("relu")(enconv_25)
    deconv_26= Convolution2D(64,3,strides=(1, 1), padding="same", kernel_initializer='he_normal',dilation_rate=(3,3))(enconv_25)
    deconv_26 = BatchNormalization()(deconv_26)
    deconv_26= Activation("relu")(deconv_26)
    deconv_27= Convolution2D(1,1,strides=(1, 1), padding="same", kernel_initializer='he_normal',)(deconv_26)
    deconv_27= Activation("relu")(deconv_27)
    output=tf.clip_by_value(deconv_27,clip_value_min=0,clip_value_max=1)
    model = keras.Model(inputs=input_images, outputs=output, name="Connected-SegNets")
    return model

model=defineNetwork()
model.summary()
model.compile(
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    metrics=['accuracy'],
)

#tf.keras.utils.plot_model(model, show_shapes=True)

#train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        #rescale=1./255,
        #shear_range=0.2,
        #zoom_range=0.2,
        #horizontal_flip=True)
#train_generator = train_datagen.flow_from_directory(
        #'D:\BreastCancer\sample',
        #target_size=(250, 250),
        #batch_size=5,
        #class_mode=None)
#print(train_generator)
#train_ds = tf.keras.utils.image_dataset_from_directory(
  #'D:\BreastCancer\sample',
  #validation_split=0.1,
  #subset="training",
  #seed=123,
  #image_size=(256, 256),
  #batch_size=5)
#test_ds=train_ds
#print(train_ds)




#test_scores = model.evaluate(x, y_test, verbose=2)
def findMammogram(path,end=1):
  obj = os.scandir(path)
  for entry in obj :
    for entries in os.scandir(entry.path):
      newPath=os.path.join(entries.path,f'1-{end}.dcm')
      print (newPath)
      return newPath
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


data = pd.read_csv(r"E:\ADMANI\manifest-ZkhPvrLo5216730872708713142\mass_case_description_train_set.csv")
base=r'e:\ADMANI\manifest-ZkhPvrLo5216730872708713142\CBIS-DDSM'
roiPath2=''
mamImages=[]
roiImages=[]
maskImages=[]

#finding all images for every patient
for x in range(50): 
  #fetch values from csv
  laterality=data._get_value(x,'left or right breast')
  view=data._get_value(x,'image view')
  id=data._get_value(x,'patient_id') 
  number=data._get_value(x,'abnormality id')
  path=''
  pathList=[]
  #create base path for full mammogram
  if number ==1:
    path=f"Mass-Training_{id}_{laterality}_{view}"
    path=os.path.join(base,path)
    mamPath=findMammogram(path)
    path2=path+'_1'
    im1=findMammogram(path2)
    im2=findMammogram(path2,2)
    if os.path.getsize(im1)>os.path.getsize(im2):
        roiPath=im2
        maskPath=im1
    else:
      roiPath=im1
      maskPath=im2
    pathList=[mamPath,roiPath,maskPath]
  #add multiple ROI's from same patient(if)
  if number!=1:
    path=f"Mass-Training_{id}_{laterality}_{view}"
    path=os.path.join(base,path)
    newPath= path + f'_{number}'
    im1=findMammogram(newPath)
    im2=findMammogram(newPath,2)
    if os.path.getsize(im1)>os.path.getsize(im2):
      roi=im2
      mask=im1
    else:
      roi=im1
      mask=im2
    pathList.append(roi)
    pathList.append(mask)
  for i in pathList:
    ds= dicom.dcmread(i)
    # Convert to float to avoid overflow or underflow losses.
    new_image = ds.pixel_array.astype(float)
    # Rescaling grey scale between 0-255
    scaled_image = (np.maximum(new_image, 0) / new_image.max()) * 255.0
    # Convert to uint
    scaled_image = np.uint16(scaled_image)
    image=tf.convert_to_tensor(scaled_image,dtype=tf.uint16)
    image=tf.expand_dims(image,-1)
    image=tf.expand_dims(image,0)
    if len(pathList)==3:
      if pathList.index(i)==0:
        mamImages.append(image)
      elif pathList.index(i)==1:
        roiImages.append(image)
      else:
        maskImages.append(image)
    elif len(pathList)==2:
      if pathList.index(i)==0:
        roiImages.append(image)
      elif pathList.index(i)==1:
        maskImages.append(image)
x = [resize_image(i) for i in mamImages]
y = [resize_mask(m) for m in maskImages]
x=tf.data.Dataset.from_tensor_slices(x)
y=tf.data.Dataset.from_tensor_slices(y)
train=tf.data.Dataset.zip((x, y))
print(x,y)

  
history = model.fit(train, epochs=20,batch_size=4)
#final_image = Image.fromarray(scaled_image)
#final_image=cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
#ray_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)

#ret,thresh = cv2.threshold(gray_image,127,255,0)
#contours,hierarchy = cv2.findContours(thresh, 1, 2)
#cnt = contours[0]

#x,y,w,h = cv2.boundingRect(cnt)
#cv2.rectangle(final_image,(x,y),(x+w,y+h),(0,255,0),2)
#cv2.imshow('Bounding Box',final_image)
#plt.imshow(final_image)
#plt.show()
