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

def plot_data(data, kind):
    plt.figure()
    plt.hist(data, bins = range(n_classes + 1))
    plt.title('Class distribution of {} Data'.format(kind))
    plt.xlabel('Classes')
    plt.ylabel('Count')
    ax = plt.gca()
    ax.set_fc('k')
    plt.savefig('{}_data_viz.png'.format(kind))


for data, kind in zip([y_train, y_valid, y_test], ['training', 'validation', 'testing']):
    plot_data(data, kind)


# Check pixel distribution across each label
import os
from functools import reduce

ROT_DIR = 'rotational_distrib'

def get_avg_image_per_label(label):
    idx = y_train == label
    X_avg = np.mean(X_train[idx], axis = 0)
    #filename = os.path.join(RND_DIR, 'avg_img_{}.png'.format(CLASS_MAPPER[label]))
    filename = os.path.join(RND_DIR, 'avg_img_{}.png'.format(label))
    mpimg.imsave(filename, X_avg)

for label in range(n_classes):
    get_avg_image_per_label(label)

# find outliers
# check avg pixel intensity of each image in training set

for i in range(n_train):
    pass























