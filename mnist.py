import gzip
import pickle
from urllib.request import urlretrieve
import zipfile

import nengo
import nengo_dl
import tensorflow as tf
import numpy as np
import matplotlib as plt
plt.use('Agg')


#urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", "mnist.pkl.gz")
with gzip.open("mnist.pkl.gz") as f:
    train_data, _, test_data = pickle.load(f, encoding="latin1")
print(len(train_data[0][2]))
print (type(train_data))
train_data = list(train_data) #50000 images
print (type(train_data))
#print(train_data[0][0])
test_data = list(test_data) #10000 images
# for data in (train_data, test_data): 
# 	#data[0] are images, stored as length 28*28=784 arrays
# 	#data[1] are lables, stored in a length 50000 or 10000 integer array
# 	one_hot = np.zeros((data[0].shape[0], 10)) 
# 	one_hot[np.arange(data[0].shape[0]), data[1]] = 1 #convert labels into a length 10 vector
# 	data[1] = one_hot

# # for i in range(3):
# #     plt.pyplot.figure()
# #     plt.pyplot.imshow(np.reshape(train_data[0][i], (28, 28)))
# #     plt.pyplot.axis('off')
# #     plt.pyplot.title(str(np.argmax(train_data[1][i])));
# #     plt.pyplot.show()

# #construct network architecture suitable for MNIST
# with nengo.Network() as net:
# 	# set some default parameters for the neurons that will make
# 	# the training progress more smoothly
# 	net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
# 	net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
# 	neuron_type = nengo.LIF(amplitude=0.01)

# 	# we'll make all the nengo objects in the network
# 	# non-trainable. we could train them if we wanted, but they don't
# 	# add any representational power. note that this doesn't affect
# 	# the internal components of tensornodes, which will always be
# 	# trainable or non-trainable depending on the code written in
# 	# the tensornode.
# 	nengo_dl.configure_settings(trainable=False)

# 	# the input node that will be used to feed in input images
# 	inp = nengo.Node([0] * 28 * 28)

# 	# add the first convolutional layer
# 	x = nengo_dl.tensor_layer(
# 	    inp, tf.layers.conv2d, shape_in=(28, 28, 1), filters=32,
# 	    kernel_size=3)

# 	# apply the neural nonlinearity
# 	x = nengo_dl.tensor_layer(x, neuron_type)

# 	# add another convolutional layer
# 	x = nengo_dl.tensor_layer(
# 	    x, tf.layers.conv2d, shape_in=(26, 26, 32),
# 	    filters=64, kernel_size=3)
# 	x = nengo_dl.tensor_layer(x, neuron_type)

# 	# add a pooling layer
# 	x = nengo_dl.tensor_layer(
# 	    x, tf.layers.average_pooling2d, shape_in=(24, 24, 64),
# 	    pool_size=2, strides=2)

# 	# another convolutional layer
# 	x = nengo_dl.tensor_layer(
# 	        x, tf.layers.conv2d, shape_in=(12, 12, 64),
# 	        filters=128, kernel_size=3)
# 	x = nengo_dl.tensor_layer(x, neuron_type)

# 	# another pooling layer
# 	x = nengo_dl.tensor_layer(
# 	    x, tf.layers.average_pooling2d, shape_in=(10, 10, 128),
# 	    pool_size=2, strides=2)

# 	# linear readout
# 	x = nengo_dl.tensor_layer(x, tf.layers.dense, units=10)

# 	# we'll create two different output probes, one with a filter
# 	# (for when we're simulating the network over time and
# 	# accumulating spikes), and one without (for when we're
# 	# training the network using a rate-based approximation)
# 	out_p = nengo.Probe(x)
# 	out_p_filt = nengo.Probe(x, synapse=0.1)

# #create simulator object
# minibatch_size = 200
# sim = nengo_dl.Simulator(net, minibatch_size=minibatch_size)


# # note that we need to add the time dimension (axis 1), which will be a
# # single step during training

train_inputs = {inp: train_data[0][:, None, :]}
train_targets = {out_p: train_data[1][:, None, :]}

# when testing our network with spiking neurons we will need to run it over time,
# so we repeat the input/target data for a number of timesteps. we're also going
# to reduce the number of test images, just to speed up this example.
n_steps = 30
test_inputs = {inp: np.tile(test_data[0][:minibatch_size*2, None, :],
                            (1, n_steps, 1))}
test_targets = {out_p_filt: np.tile(test_data[1][:minibatch_size*2, None, :],
                                    (1, n_steps, 1))}

#define error function
def objective(x, y):
	return tf.nn.softmax_cross_entropy_with_logits_v2(logits=x, labels=y)

#define optimizer
opt = tf.train.RMSPropOptimizer(learning_rate=0.001)

def classification_error(outputs, targets):
	return 100 * tf.reduce_mean(tf.cast(tf.not_equal(tf.argmax(outputs[:, -1], axis=-1), tf.argmax(targets[:, -1], axis=-1)), tf.float32))

print("error before training: %.2f%%" % sim.loss(test_inputs, test_targets, {out_p_filt: classification_error}))


#set this variable to True if want to perform training, else we will download some pretrained weights
do_training = False #True
if do_training:
	# run training
	sim.train(train_inputs, train_targets, opt, objective={out_p: objective}, n_epochs=10)

	# save the parameters to file
	sim.save_params("./mnist_params")
else:
	# download pretrained weights
	urlretrieve("https://drive.google.com/uc?export=download&id=18ErAZh0LFRaISLHDM-dJjjnzqvYfyjo0", "mnist_params.zip")
	with zipfile.ZipFile("mnist_params.zip") as f:
		f.extractall()

	# load parameters
	sim.load_params("./mnist_params")

# print("error after training: %.2f%%" % sim.loss(test_inputs, test_targets, {out_p_filt: classification_error}))

# n_step could be increased to further improve performance, since we would get a more accurate measure of each spiking neuron’s output.

# We can also plot some example outputs from the network, to see how it is performing over time.

# sim.run_steps(n_steps, input_feeds={inp: test_inputs[inp][:minibatch_size]})

# for i in range(5):
# 	plt.figure()
# 	plt.subplot(1, 2, 1)
# 	plt.imshow(np.reshape(test_data[0][i], (28, 28)))
# 	plt.axis('off')

# 	plt.subplot(1, 2, 2)
# 	plt.plot(sim.trange(), sim.data[out_p_filt][i])
# 	plt.legend([str(i) for i in range(10)], loc="upper left")
# 	plt.xlabel("time")

# sim.close()







