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
  r'd:\CMMd',
  validation_split=0.2,
  subset="training",
  color_mode='rgb',
  seed=120,
  image_size=(1660, 700),
  batch_size=1)     
val_ds = tf.keras.utils.image_dataset_from_directory(
  r'd:\CMMd',
  validation_split=0.2,
  subset="training",
  color_mode='rgb',
  seed=120,
  image_size=(1660, 700),
  batch_size=1) 

#AUTOTUNE = tf.data.AUTOTUNE
#train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
#val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


class_names=train_ds.class_names
#plt.figure(figsize=(10, 10))
#for images, labels in train_ds.take(1):
  #for i in range(4):
    #ax = plt.subplot(2, 2, i + 1)
    #plt.imshow(images[i].numpy().astype('uint8'))
    #plt.title(class_names[labels[i]])
    #plt.axis("off")
#plt.show()

model=tf.keras.applications.convnext.ConvNeXtSmall(
    model_name='convnext_small',
    include_top=False,
    weights='imagenet',
    #input_tensor=tf.keras.layers.Input(shape = (2294,1914,3)),
    input_shape=(1660,700,3),
    #consider tinkering with classes
    classes=3,
    classifier_activation='softmax'
)
output= tf.keras.layers.GlobalAveragePooling2D()(model.layers[-1].output)
output=tf.keras.layers.LayerNormalization()(output)
output=tf.keras.layers.Dense(1,kernel_initializer='random_uniform')(output)
#output=tf.keras.layers.Flatten()(output)

newModel=keras.Model(inputs=model.inputs, outputs=output,name='BCDConvNext')

newModel.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    metrics=[
        metrics.AUC(),
        metrics.Accuracy(),
        metrics.Precision()
    ]
)
#newModel.summary()

history=newModel.fit(
    train_ds,
    epochs=10,
    validation_data=val_ds
    )



model2=tf.keras.applications.convnext.ConvNeXtBase(
    model_name='convnext_base',
    include_top=False,
    weights='imagenet',
    input_shape=(2294,1914,3),
    pooling=None,
    classes=3,
    classifier_activation='softmax'
)
