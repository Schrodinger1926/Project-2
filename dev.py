# Load pickled data
import pickle
import os
# TODO: Fill this in based on where you saved the training and testing data

DATA_DIR = 'data'
training_file = os.path.join(DATA_DIR, 'train.p')
validation_file = os.path.join(DATA_DIR, 'valid.p')
testing_file = os.path.join(DATA_DIR, 'test.p')

with open(training_file, mode='rb') as f:
    train = pickle.load(f)

with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)

with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


#-----------------------------------------------------------

### Replace each question mark with the appropriate value. ### Rep 
### Use python, pandas or numpy methods rather than hard coding the results
import numpy as np

# TODO: Number of training examples
n_train = X_train.shape[0]

# TODO: Number of validation examples
n_validation = X_valid.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_train.shape[1:]

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))
print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

#X_train_df = np.reshape(X_train, newshape = (n_train, image_shape[0]**image_shape[1]))
#-----------------------------------------------------------

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import random

# Research folders
RND_DIR = 'rnd'
if not os.path.isdir(RND_DIR):
    os.mkdir(RND_DIR)


# Visualizations will be shown in the notebook.

CLASS_MAPPER = {}
with open('signnames.csv', 'r') as f:
    # skip first line
    lines = f.readlines()[1:]
    for line in lines:
        label_id, label_name = line.strip().split(',')
        CLASS_MAPPER[int(label_id)] = label_name

# Display any image for sanity check
idx = random.choice(range(n_train))
mpimg.imsave('sanity_check.png', X_train[idx])
#mpimg.imshow(X_train[idx])

def plot_data(data, xlabel, ylabel, kind, n_items):
    plt.figure()
    plt.hist(data, bins = range(n_items + 1))
    plt.title('Class distribution of {} Data'.format(kind))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax = plt.gca()
    ax.set_fc('k')
    plt.savefig('{}_data_viz.png'.format(kind))


for data, kind in zip([y_train, y_valid, y_test], ['training', 'validation', 'testing']):
    plot_data(data = data,
              xlabel = 'classes',
              ylabel = 'count',
              kind = 'training',
              n_items = n_classes)


# Check pixel distribution across each label
import os
import cv2
from functools import reduce

ROT_DIR = 'rotational_distrib'

def get_avg_image_per_label(label):
    idx = y_train == label
    X_avg = np.mean(X_train[idx], axis = 0).astype(dtype = np.uint8)
    filename = os.path.join(RND_DIR, 'avg_img_{}.png'.format(label))
    mpimg.imsave(filename, X_avg)
    #SID: imshow here

for label in range(n_classes):
    get_avg_image_per_label(label)

# find outliers
# check avg pixel intensity of each image in training set

import sys

color_mean = []
for i in range(n_train):
    img = X_train[i]
    color_mean.append(int(np.mean(img)))

plot_data(data = color_mean,
          xlabel = 'pixel values',
          ylabel = 'intensity',
          kind = 'training_avg_intensity',
          n_items = 256)

# find outliers


#----------------------- Pre-Processing  ------------------------------------

import cv2

def get_gray_mean_normalized(data):

    # convert to gray scale
    gray_operator = np.array([0.299, 0.587, 0.114])

    # normalize
    data_mean_norm = np.expand_dims(np.dot(data, gray_operator), axis = 3)

    assert(data_mean_norm.shape == (data.shape[0], 32, 32, 1))

    return data_mean_norm


def get_augmented_data(data):
    # salt pepper

    # fine rotation

    # 

# data augmentation
X_train = get_augmented_data(X_train)

X_train = get_gray_mean_normalized(data = X_train)
X_valid = get_gray_mean_normalized(data = X_valid)
X_test = get_gray_mean_normalized(data = X_test)



#----------------------- Model Architecture  ------------------------------------
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle

EPOCHS = 100
BATCH_SIZE = 128

def LeNet(x):
    # Hyperparameters
    mu = 0
    sigma = 0.1

    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)

    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b

    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b

    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits


x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

rate = 0.0005

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

def get_variance_bias_plot(train_data, valid_data):
    plt.figure()
    assert(len(train_data) == len(valid_data))
    n = len(train_data)

    # Plot data
    train_line, = plt.plot(range(n), train_data, 'r', label = 'Training')
    valid_line, = plt.plot(range(n), valid_data, 'b', label = 'Validation')
    plt.title('Variance-Bias analysis')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    # add legend
    first_legend = plt.legend(handles=[train_line], loc = 4)
    plt.gca().add_artist(first_legend)
    plt.legend(handles=[valid_line], loc = 3)

    # set background color
    plt.gca().set_fc('k')

    plt.savefig('variance_bias_iter_2.png'.format(kind))


train_acc_list, valid_acc_list = [], []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            assert(batch_x.shape[0] == batch_y.shape[0])
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        training_accuracy = evaluate(X_train, y_train)
        validation_accuracy = evaluate(X_valid, y_valid)
        # collect data
        train_acc_list.append(training_accuracy)
        valid_acc_list.append(validation_accuracy)

        print("EPOCH {} ...".format(i+1))
        print("Training Accuracy = {:.3f} | Validation Accuracy = {:.3f}".format(training_accuracy, validation_accuracy))
        print()
    print("Getting training analysis")
    get_variance_bias_plot(train_acc_list, valid_acc_list)
    saver.save(sess, './lenet')
    print("Model saved")


