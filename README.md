# Project
Simplilearn project
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import division, print_function, unicode_literals
import os
import cv2
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from PIL import Image
from random import shuffle
from tqdm import tqdm
import random
import sys
%matplotlib inline

# import the Keras Libaries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils


# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals


# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    
    # set Hyperparameter

reset_graph()

img_size = 32
num_channels = 3
img_size_flat = img_size * img_size * num_channels
img_shape = (img_size, img_size)
trainpath = ('C:\\Users\HI\Desktop\A.I course\Deep Learning course - Capstone Project\Deep Learning course - Capstone Project\data\train')
testpath = ('C:\\Users\HI\Desktop\A.I course\Deep Learning course - Capstone Project\Deep Learning course - Capstone Project\data\test')
labels = {'cats': 0, 'dogs': 1}
fc_size=32 #size of the output of final FC layer
num_steps=100 #Try 100, 200, 300. number of steps that training data should be looped. Usually 20K
tf.logging.set_verbosity(tf.logging.INFO)


def read_images_classes(basepath,imgSize=img_size):
    image_stack = []
    label_stack = []

    for counter, l in enumerate(labels):
        path = os.path.join(basepath, l,'*g')
        for img in glob.glob(path):
            one_hot_vector =np.zeros(len(labels),dtype=np.int16)
            one_hot_vector[counter]=1
            image = cv2.imread(img)
            im_resize = cv2.resize(image,img_shape, interpolation=cv2.INTER_CUBIC)
            image_stack.append(im_resize)
            label_stack.append(labels[l])            
    return np.array(image_stack), np.array(label_stack)

X_train, y_train=read_images_classes(trainpath)
X_test, y_test=read_images_classes(testpath)

#test a sample image
print('length of train image set',len(X_train))
print('X_data shape:', X_train.shape)
print('y_data shape:', y_train.shape)

fig1 = plt.figure() 
ax1 = fig1.add_subplot(2,2,1) 
img = cv2.resize(X_train[0],(img_size,img_size), interpolation=cv2.INTER_CUBIC)
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title(y_train[0])
plt.show()

# This should be done by the student
def cnn_model_fn(features, labels, mode):
    model=Sequential()
    model.add(Conv2D(32,(5,5),input_shape=(1,28,28),activation='relu'))#convolutional layer with 32 feature
    model.add(MaxPooling2D(pool_size=(2,2)))#Pooling layer with size 2*2
    model.add(Conv2D(64,(5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    model.add(Flatten())# Converting 2d matrix into 1D array
    model.add(Dense(32,activation='relu'))
    model.add(Dense(num_classes,activation="softmax"))
    model.compile(optimizer="adam",metrics=["accuracy"],loss="categorical_crossentropy")
    return(model)
    
    #X_train = np.array((X_train/255.0),dtype=np.float16)
#X_test = np.array((X_test/255.0), dtype=np.float16)
X_train = np.array((X_train/255.0),dtype=np.float32)
X_test = np.array((X_test/255.0), dtype=np.float32)

pets_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/tmp/pets_convnet_model")
#pets_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn)
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": X_train}, y=y_train, batch_size=10,
                                                      num_epochs=None, shuffle=True)
pets_classifier.train(input_fn=train_input_fn, steps=num_steps, hooks=[logging_hook])
eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": X_test}, y=y_test, num_epochs=1,shuffle=False)
eval_results = pets_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)
