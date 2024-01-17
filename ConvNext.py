import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom as pydicom
import tensorflow as tf
from PIL import Image
import cv2
from tensorflow import keras
from keras import metrics
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

#config = ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.8
#session = InteractiveSession(config=config)

gpu = tf.config.list_physical_devices('GPU')
print(gpu)
for i in gpu:
    tf.config.experimental.set_memory_growth(i, True)
    

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
def findFiles(path):
    paths=[]
    for subdir,dir, files in os.walk(path):
        for file in files:
            paths.append(os.path.join(subdir,file))
            print(file)
    #print(paths)
    return paths

cancerData = pd.read_excel(r"F:\cmmd\manifest-1616439774456\CMMD_clinicaldata_revision.xlsx")  
filePaths=pd.read_csv(r"F:\cmmd\manifest-1616439774456\metadata.csv")     
base=r"f:\cmmd\manifest-1616439774456\CMMD"
lastid=0
for i in range(0):
   break
   cancImages=[]
   normalImages=[]
   id=cancerData._get_value(i,'ID1')
   nextid=cancerData._get_value(i+1,'ID1')
   classValue=cancerData._get_value(i,'classification')
   laterality=cancerData._get_value(i,'LeftRight')
   print(laterality)
   path=os.path.join(base,id)
   temp=findFiles(path)

   if len(temp)==2:
    cancImages=temp
   elif laterality=='R' and len(temp)==4 and (nextid==id or lastid==id):
    cancImages=temp[2:]
    lastid=id
   elif laterality=='L' and len(temp)==4 and (nextid==id or lastid==id):
    cancImages=temp[0:2]
    lastid=id
   elif laterality=='L' and len(temp)==4 and nextid!=id:
    cancImages=temp[0:2]
    normalImages=temp[2:]
   elif laterality=='R' and len(temp)==4 and nextid!=id:
    cancImages=temp[2:]
    normalImages=temp[0:2]
   
   #print(cancImages)
   #print(normalImages) 

   for s in cancImages:
    image = get_pixels_with_windowing(s)
    image=Image.fromarray(image.astype('uint8'), 'L')
    image.save(f'd:\CMMd\{classValue}\{id+"_"+str(temp.index(s)+1)}.png')
   for s in normalImages:
    image = get_pixels_with_windowing(s)
    image=Image.fromarray(image.astype('uint8'), 'L')
    y=os.path.join(r'd:\CMMd\Normal',f'{id+"_"+str(temp.index(s)+1)}.png')
    image.save(y)


train_ds = tf.keras.utils.image_dataset_from_directory(
  '/content/drive/MyDrive/Vinddrextractions',
  validation_split=0.15,
  label_mode='categorical',
  subset="training",
  color_mode='rgb',
  seed=120,
  image_size=(1711, 940),
  batch_size=1)
val_ds = tf.keras.utils.image_dataset_from_directory(
  '/content/drive/MyDrive/Vinddrextractions',
  validation_split=0.15,
  subset="validation",
  label_mode='categorical',
  color_mode='rgb',
  seed=120,
  image_size=(1711, 940),
  batch_size=1)

data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(1571,
                                  1052,
                                  3)),
    layers.RandomRotation(0.22),
    layers.RandomZoom(0.1),
  ]
)
class_names=train_ds.class_names
#train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))
#val_ds = val_ds.map(lambda x, y: (data_augmentation(x), y))


model=tf.keras.applications.convnext.ConvNeXtSmall(
    model_name='convnext_small',
    include_top=False,
    weights='imagenet',
    #input_tensor=tf.keras.layers.Input(shape = (2294,1914,3)),
    input_shape=(1711,940,3)
)
output= tf.keras.layers.GlobalAveragePooling2D()(model.layers[-1].output)
output=tf.keras.layers.LayerNormalization()(output)
output=tf.keras.layers.Dense(3,kernel_initializer='random_uniform',activation='softmax')(output)

initial_learning_rate = 0.0001
final_learning_rate = 0.000005
epochs=6
learning_rate_decay_factor = (final_learning_rate / initial_learning_rate)**(1/epochs)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=initial_learning_rate,
                decay_steps=20771,
                decay_rate=learning_rate_decay_factor,
                staircase=True)


newModel=keras.Model(inputs=model.inputs, outputs=output,name='BCDConvNext')
newModel.load_weights('/content/drive/MyDrive/cp-0001.ckpt')
newModel.compile(
    loss=keras.losses.CategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(learning_rate=0.00001),
    metrics=[
        metrics.AUC(),
        metrics.CategoricalAccuracy(),
        metrics.Precision()
    ]
)
checkpoint_path = "/content/drive/MyDrive/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq=20771)
history2=newModel.fit(
    train_ds,
    epochs=15,
    validation_data=val_ds,
    callbacks=[cp_callback]
    )
newModel.save_weights('/content/drive/MyDrive/finalweights3.ckpt')
