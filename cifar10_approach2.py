
# Use GPU for Theano, comment to use CPU instead of GPU
# Tensorflow uses GPU by default
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

# If using tensorflow, set image dimensions order
from keras import backend as K
if K.backend()=='tensorflow':
	K.set_image_dim_ordering("th")

import pickle
import time
import numpy as np
np.random.seed(2017) 
# import matplotlib
# # matplotlib.use('Agg')
# import matplotlib.pyplot as plt

import tensorflow as tf 
import nengo
import nengo_dl

# from keras.models import Sequential
# from keras.layers.convolutional import Convolution2D, MaxPooling2D
# from keras.layers import Activation, Flatten, Dense, Dropout
# from keras.layers.normalization import BatchNormalization
# from keras.utils import np_utils
# from keras.datasets.cifar import load_batch

# num_train_samples = 20000

# pre-download data/test batches to "path"
# path = "/Users/rw/Desktop/ctn01/cifar-10-batches-py"
path = "/home/s672wang/cifar10/cifar-10-batches-py"

with open(path + '/data_batch_' + '1', mode='rb') as file:
	# note the encoding type is 'latin1'
	batch = pickle.load(file, encoding='latin1')  
train_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1).reshape(len(batch['data']), 3072)
train_labels = batch['labels']

# for batch_id in range(2,6):
# 	with open(path + '/data_batch_' + str(batch_id), mode='rb') as file:
# 		# note the encoding type is 'latin1'
# 		batch = pickle.load(file, encoding='latin1')  
# 	temp = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1).reshape(len(batch['data']), 3072)
# 	train_features = np.append(train_features, temp, axis=0)
# 	temp = batch['labels']
# 	train_labels.extend(temp)

encoded = np.zeros((len(train_labels), 10))
for idx, val in enumerate(train_labels):
    encoded[idx][val] = 1
train_labels = encoded

with open(path + '/test_batch', mode='rb') as file:
		# note the encoding type is 'latin1'
		batch = pickle.load(file, encoding='latin1')
        
test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1).reshape(len(batch['data']), 3072)
test_labels = batch['labels']

encoded = np.zeros((len(test_labels), 10))
for idx, val in enumerate(test_labels):
    encoded[idx][val] = 1
test_labels = encoded

#0airplane, 1automobile, 2bird, 3cat, 4deer, 5dog, 6frog, 7horse, 8ship, 9truck
# for i in range(1,800,80):
#     plt.figure()
#     plt.imshow(np.reshape(train_features[i], (32, 32,3)))
#     plt.axis('off')
#     plt.title(str(np.argmax(train_labels[i])));
#     plt.show()

# num_train, img_channels, img_rows, img_cols =  train_features.shape
# num_test, _, _, _ =  test_features.shape
# num_classes = len(np.unique(train_labels))

# #pre-processing 
# train_features = train_features.astype('float32')/255
# test_features = test_features.astype('float32')/255
# # convert class labels to binary class labels
# train_labels = np_utils.to_categorical(train_labels, num_classes)
# test_labels = np_utils.to_categorical(test_labels, num_classes)

with nengo.Network() as net:
	neuron_type = nengo.LIF(amplitude=0.001)
	inp = nengo.Node([0]*3*32*32)
	nengo_dl.configure_settings(trainable=False)

	x = nengo_dl.tensor_layer(inp, tf.layers.conv2d, shape_in=(32,32,3), filters=32, kernel_size=3, use_bias=True)
	x = nengo_dl.tensor_layer(x, neuron_type)

	x = nengo_dl.tensor_layer(x, tf.layers.dropout, rate=0.1)

	x = nengo_dl.tensor_layer(x, tf.layers.conv2d,shape_in=(30,30,32),filters=64, kernel_size=5, strides=2, use_bias=False)
	x = nengo_dl.tensor_layer(x, neuron_type)

	x = nengo_dl.tensor_layer(x, tf.layers.dropout,rate=0.2)

	x = nengo_dl.tensor_layer(x, tf.layers.flatten)

	x = nengo_dl.tensor_layer(x, tf.layers.dense, units=128)
	x = nengo_dl.tensor_layer(x, neuron_type)

	x = nengo_dl.tensor_layer(x, tf.layers.dropout,rate=0.3)

	x = nengo_dl.tensor_layer(x, tf.layers.dense, units=10,activation=tf.keras.activations.softmax)

	out_p = nengo.Probe(x)
	out_p_filt = nengo.Probe(x, synapse=0.1)

# with nengo.Network() as net:
# 	net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
# 	net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
# 	neuron_type = nengo.LIF(amplitude=0.01)

# 	nengo_dl.configure_settings(trainable=False)

# 	inp = nengo.Node([0]*3*32*32)

# 	#1st conv layer
# 	x = nengo_dl.tensor_layer(inp, tf.layers.conv2d, shape_in=(32,32,3),filters=48, kernel_size=3, padding='same')
# 	#non-linearity
# 	x = nengo_dl.tensor_layer(x, neuron_type)

# 	#2nd conv layer
# 	x = nengo_dl.tensor_layer(x, tf.layers.conv2d,shape_in=(32,32,48),filters=48, kernel_size=3)
# 	x = nengo_dl.tensor_layer(x, neuron_type)

# 	#1st pooling layer
# 	x = nengo_dl.tensor_layer(x, tf.layers.average_pooling2d,shape_in=(30,30,48),pool_size=2,strides=2) #strides 2? 
# 	x = nengo_dl.tensor_layer(x, tf.layers.dropout, rate=0.25)

# 	#3rd conv layer
# 	x = nengo_dl.tensor_layer(x, tf.layers.conv2d,shape_in=(15,15,48),filters=96,kernel_size=3, padding='same')
# 	x = nengo_dl.tensor_layer(x, neuron_type)

# 	#4th conv layer
# 	x = nengo_dl.tensor_layer(x, tf.layers.conv2d,shape_in=(15,15,96),filters=96,kernel_size=3)
# 	x = nengo_dl.tensor_layer(x, neuron_type)

# 	#2nd pooling layer
# 	x = nengo_dl.tensor_layer(x, tf.layers.average_pooling2d, shape_in=(13,13,96),pool_size=2, strides=2) #strides2?
# 	x = nengo_dl.tensor_layer(x, tf.layers.dropout, rate=0.25)

# 	#5th conv layer
# 	x = nengo_dl.tensor_layer(x, tf.layers.conv2d, shape_in=(6,6,96),filters=192, kernel_size=3, padding='same')
# 	x = nengo_dl.tensor_layer(x, neuron_type)

# 	#6th conv layer
# 	x = nengo_dl.tensor_layer(x, tf.layers.conv2d, shape_in=(6,6,192),filters=192, kernel_size=3)
# 	x = nengo_dl.tensor_layer(x, neuron_type)

# 	#3rd pooling layer
# 	x = nengo_dl.tensor_layer(x, tf.layers.average_pooling2d,shape_in=(4,4,192),pool_size=2,strides=2)
# 	x = nengo_dl.tensor_layer(x, tf.layers.dropout, rate=0.25)

# 	x = nengo_dl.tensor_layer(x, tf.layers.flatten) #ther's also Flatten?

# 	# linear readout
# 	x = nengo_dl.tensor_layer(x, tf.layers.dense, units=512)
# 	x = nengo_dl.tensor_layer(x, neuron_type)
# 	x = nengo_dl.tensor_layer(x, tf.layers.dropout, rate=0.5)
# 	x = nengo_dl.tensor_layer(x, tf.layers.dense, units=256)
# 	x = nengo_dl.tensor_layer(x, neuron_type)
# 	x = nengo_dl.tensor_layer(x, tf.layers.dropout, rate=0.5)
# 	x = nengo_dl.tensor_layer(x, tf.layers.dense, units=10,activation=tf.keras.activations.softmax)


# 	out_p = nengo.Probe(x)
# 	out_p_filt = nengo.Probe(x, synapse=0.1)



minibatch_size = 200
sim = nengo_dl.Simulator(net, minibatch_size=minibatch_size,tensorboard=path+"/loss")

n_steps = 30

train_inputs = {inp: train_features[:, None, :]}
train_targets = {out_p: train_labels[:, None, :]}

overfitting_in = {inp: np.tile(train_features[:minibatch_size*50:10, None, :], (1, n_steps, 1))}
overfitting_target = {out_p_filt: np.tile(train_labels[:minibatch_size*50:10, None, :], (1, n_steps, 1))}

test_inputs = {inp: np.tile(test_features[:minibatch_size*5, None, :], (1, n_steps, 1))}
test_targets = {out_p_filt: np.tile(test_labels[:minibatch_size*5, None, :], (1, n_steps, 1))}

opt = tf.train.AdamOptimizer()

def objective (x, y):
	return tf.nn.softmax_cross_entropy_with_logits_v2(logits=x, labels=y)

def classification_error(outputs, targets):
	return 100 * tf.reduce_mean(tf.cast(tf.not_equal(tf.argmax(outputs[:, -1], axis=-1), tf.argmax(targets[:, -1], axis=-1)), tf.float32))

print("error before training: %.2f%%" % sim.loss(test_inputs, test_targets, {out_p_filt: classification_error}))

sim.train(train_inputs, train_targets, opt, objective={out_p: objective}, n_epochs=200, summaries=["loss"])

sim.save_params("/home/s672wang/cifar10/cifar10_params")

# compute test accuracy
print("error after training: %.2f%%" % sim.loss(test_inputs, test_targets, {out_p_filt: classification_error}))
print("error after training on training set : %.2f%%" % sim.loss(overfitting_in, overfitting_target, {out_p_filt: classification_error}))

sim.close()


##########Original:: Define the model##################
#model = Sequential()
#model.add(Convolution2D(48, 3, 3, border_mode='same', input_shape=(3, 32, 32)))
#model.add(Activation('relu'))
#model.add(Convolution2D(48, 3, 3))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#model.add(Convolution2D(96, 3, 3, border_mode='same'))
#model.add(Activation('relu'))
# model.add(Convolution2D(96, 3, 3))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Convolution2D(192, 3, 3, border_mode='same'))
# model.add(Activation('relu'))
# model.add(Convolution2D(192, 3, 3))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(256))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))
# Compile the model

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#model_info = model.fit(train_features, train_labels, batch_size=128, nb_epoch=200, validation_data = (test_features, test_labels), verbose=0)
###############################################################

# Train the model
# print ("Accuracy on test data before training is: %0.2f"%accuracy(test_features, test_labels, model))
#----------

# plot model history
#plot_model_history(model_info)

# print ("Accuracy on test data after training is: %0.2f"%accuracy(test_features, test_labels, model))


# # from keras.preprocessing.image import ImageDataGenerator

# # datagen = ImageDataGenerator(zoom_range=0.2, 
# #                             horizontal_flip=True)


# # # train the model
# # start = time.time()
# # # Train the model
# # model_info = model.fit_generator(datagen.flow(train_features, train_labels, batch_size = 128),
# #                                  samples_per_epoch = train_features.shape[0], nb_epoch = 200, 
# #                                  validation_data = (test_features, test_labels), verbose=0)
# # end = time.time()
# # print "Model took %0.2f seconds to train"%(end - start)
# # # plot model history
# # plot_model_history(model_info)
# # # compute test accuracy
# # print "Accuracy on test data is: %0.2f"%accuracy(test_features, test_labels, model)
