
# Use GPU for Theano, comment to use CPU instead of GPU
# Tensorflow uses GPU by default
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

# If using tensorflow, set image dimensions order
from keras import backend as K
if K.backend()=='tensorflow':
	K.set_image_dim_ordering("th")

import time
#import matplotlib.pyplot as plt
import numpy as np
#% matplotlib inline
np.random.seed(2017) 

import tensorflow as tf 
import nengo
import nengo_dl

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.datasets.cifar import load_batch

num_train_samples = 50000

# path = "/Users/rw/Desktop/ctn01/cifar-10-batches-py"
path = "/home/s672wang/cifar10/cifar-10-batches-py"
train_features = np.empty((num_train_samples, 3, 32,32), dtype='uint8')
train_labels = np.empty((num_train_samples,), dtype='uint8')

for i in range(1, 6):
    fpath = os.path.join(path, 'data_batch_' + str(i))
    (train_features[(i - 1) * 10000: i * 10000, : , : , : ],
     train_labels[(i - 1) * 10000: i * 10000]) = load_batch(fpath)

fpath = os.path.join(path, 'test_batch')
test_features, test_labels = load_batch(fpath)

train_labels = np.reshape(train_labels, (len(train_labels), 1))
test_labels = np.reshape(test_labels, (len(test_labels), 1))

# if K.image_data_format() == 'channels_last':
# 	train_features = train_features.transpose(0, 2, 3, 1)
# 	test_features = test_features.transpose(0, 2, 3, 1)
# from keras.datasets import cifar10
# (train_features, train_labels), (test_features, test_labels) = cifar10.load_data()


num_train, img_channels, img_rows, img_cols =  train_features.shape
num_test, _, _, _ =  test_features.shape
num_classes = len(np.unique(train_labels))

#show example from each class
# class_names = ['airplane','automobile','bird','cat','deer',
#                'dog','frog','horse','ship','truck']
# fig = plt.figure(figsize=(8,3))
# for i in range(num_classes):
#     ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
#     idx = np.where(train_labels[:]==i)[0]
#     features_idx = train_features[idx,::]
#     img_num = np.random.randint(features_idx.shape[0])
#     im = np.transpose(features_idx[img_num,::], (1, 2, 0))
#     ax.set_title(class_names[i])
#     plt.imshow(im)
# plt.show()


#pre-processing 
train_features = train_features.astype('float32')/255
test_features = test_features.astype('float32')/255
# convert class labels to binary class labels
train_labels = np_utils.to_categorical(train_labels, num_classes)
test_labels = np_utils.to_categorical(test_labels, num_classes)

#plot model accuracy 
# def plot_model_history(model_history):
#     fig, axs = plt.subplots(1,2,figsize=(15,5))
#     # summarize history for accuracy
#     axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
#     axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
#     axs[0].set_title('Model Accuracy')
#     axs[0].set_ylabel('Accuracy')
#     axs[0].set_xlabel('Epoch')
#     axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
#     axs[0].legend(['train', 'val'], loc='best')
#     # summarize history for loss
#     axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
#     axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
#     axs[1].set_title('Model Loss')
#     axs[1].set_ylabel('Loss')
#     axs[1].set_xlabel('Epoch')
#     axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
#     axs[1].legend(['train', 'val'], loc='best')
#     plt.show()

#compute test accuracy
def accuracy(test_x, test_y, model):
    result = model.predict(test_x)
    predicted_class = np.argmax(result, axis=1)
    true_class = np.argmax(test_y, axis=1)
    num_correct = np.sum(predicted_class == true_class) 
    accuracy = float(num_correct)/result.shape[0]
    return (accuracy * 100)


##########Original:: Define the model##################
model = Sequential()
model.add(Convolution2D(48, 3, 3, border_mode='same', input_shape=(3, 32, 32)))
model.add(Activation('relu'))
model.add(Convolution2D(48, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Convolution2D(96, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(96, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Convolution2D(192, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(192, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print ("Accuracy on test data before training is: %0.2f"%accuracy(test_features, test_labels, model))

# Train the model
model_info = model.fit(train_features, train_labels, batch_size=128, nb_epoch=200, validation_data = (test_features, test_labels), verbose=0)

# plot model history
plot_model_history(model_info)

print ("Accuracy on test data after training is: %0.2f"%accuracy(test_features, test_labels, model))


# # from keras.preprocessing.image import ImageDataGenerator

# # datagen = ImageDataGenerator(zoom_range=0.2, 
# #                             horizontal_flip=True)


# # train the model
# start = time.time()
# # Train the model
# model_info = model.fit_generator(datagen.flow(train_features, train_labels, batch_size = 128),
#                                  samples_per_epoch = train_features.shape[0], nb_epoch = 200, 
#                                  validation_data = (test_features, test_labels), verbose=0)
# end = time.time()
# print "Model took %0.2f seconds to train"%(end - start)
# # plot model history
# plot_model_history(model_info)
# # compute test accuracy
# print "Accuracy on test data is: %0.2f"%accuracy(test_features, test_labels, model)