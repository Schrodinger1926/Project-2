# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
  * Explore, summarize and visualize the data set
  * Design, train and test a model architecture
  * Use the model to make predictions on new images
  * Analyze the softmax probabilities of the new images
  * Summarize the results with a written report


  [//]: # (Image References)

    [image1]: ./assets/training_data_viz.png "Traning Visualization"
    [image2]: ./assets/validation_data_viz.png "Validation Visualization"
    [image3]: ./assets/testing_data_viz.png "Testing Visualization"

    [image4]: ./rnd/avg_img_2.png "Avg Image over all images in Speed limit (50km/h)"
    [image5]: ./rnd/avg_img_28.png "Avg Image over all images in Children crossing"

    [image6]: ./assets/training_avg_intensity_data_viz.png "Avg Pixel intensity Distribution"
    [image7]: ./examples/grayscale.jpg "Grayscaling"
    [image8]: ./examples/random_noise.jpg "Random Noise"

    [image9]: ./web_test/placeholder.png "Traffic Sign 1"
    [image10]: ./web_test/placeholder.png "Traffic Sign 2"
    [image11]: ./web_test/placeholder.png "Traffic Sign 3"
    [image12]: ./web_test/placeholder.png "Traffic Sign 4"
    [image12]: ./web_test/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

    ---


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

    I used the numpy library to calculate summary statistics of the traffic
    signs data set:

    * The size of training set is 34799
    * The size of the validation set is 12630
  * The size of test set is ?
* The shape of a traffic sign image is (32, 32 3)
  * The number of unique classes/labels in the data set is 4

#### 2. Include an exploratory visualization of the dataset.

  The following is the data set distribution over all classes/signs.

##### Distribution

  I used numpy histogram to get class distribution.

  ![alt text][image1]
  ![alt text][image2]
  ![alt text][image3]

##### Rotation

  One thing I was curious about was what if some images are rotated by some angle.
  This was important because CNN's are translation invariant but not rotational.

  So, I averaged over all the images per label.

  Result
  ![alt text][image4]

  ![alt text][image5]

  I seems there aren't much rotated images present.

  Reason being, if the there were to be enough number of rotated images at different angles then the average image per lable produces was to be extremely hazzy. 

##### Pixel Intensity distribution

  CELL: 

  Next thing I was curious about was what kind of light variation is present in data set.
  One way I though was to somehow capture the count for each pixel intensity. 

  For that I averaged every image on all three channels, put mean in an array. Finally I was left with an array where element corresponds to avg pixel value of an image. Size of array was that of the training dataset.

  Drew histogram of each pixel value count.
  ![alt text][image6]



### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


##### 1. Skewed Data
  Another thing I noticed is that training data was skewed, So I decided to roughly even out distribution.

  Options:

  1. Limit Data <br>
  Motive was to remove extra data points from some classes so as to roughly level up number of points in each class.

  I wrote a routine `get_eroded_data(data, labels, max_label_count = 2000)`, 
  where data, label are usual symbols and max_label is maximum number of points allowed in any class.

  Problem was I involved loosing vital information, I might help out in some cases, but I observed increase in loss.


  2. Augment Data

  Here I decided to generate some fake data by add some gaussian noisy data in each class in order to level up distribution. Gaussian won't add any kind of pattern in the dataset which might lead to overfitting rather it would introduce slight bias which I needed here (explained later).

  I wrote a routine `get_augmented_data(data, labels, req_size = 1000)`, 

  This further calls a function `add_gaussian_noise(data)` where data is usual symbol.

  where data, label are usual symbols and req_size tell the minimum size of each classes required. So basically it will try keep adding noise to images untill req_size per class is met.

  After some experiments I decided to go with Augment Data. Nonetheless, it didn't gave any noticable benifit in accuracy.


  Here is an example of an original image and an augmented image:

##### 2. Grayscale

  I decided to convert the images to grayscale because only a few colors (RGB) were available across the whole dataset like red, black, white. So color here doesn't really provide much information here, Moreover that may lead to overfitting.


  Here is an example of a traffic sign image before and after grayscaling.

  ![alt text][image2]



##### 3. Normalization

  Dataset was zero mean centered and scaled using `(pixel - 128)/ 128`.
  Reason being, 

  * adjust for itensity variation across images.

  * it would help in quick convergence of gradient descend due to numerical stability. Accuracy remain stuck to 70% without data normalization.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

  My final model consisted of the following layers:



  | Layer                 |     Description                                | 
  |:---------------------:|:---------------------------------------------:| 
  | Input                 | 32x32x1 RGB image,             |
  | Convolution 5x5         | 1x1 stride, valid padding, outputs 28x28x6
  |   RELU                | input 28x28x6, output 28x28x6                 |
  |  Max Pooling               | 2x2 stride, inputs 28x28x6, outputs 14x14x6   |
  | Convolution 5x5         | 1x1 stride, valid padding, outputs 10x10x16   |
  | RELU                    | input 10x10x16, output 10x10x16               |
  | Max Pooling                | 1x1 stride, inputs 10x10x16, outputs 5x5x16   |
  | Flatten                | input 5x5x16, output 400                         |
  | Fully Connected       | input 400, output 120                         |
  | RELU                    | input 120, output 120                         |
  | Dropout                | Keep probability 0.4
  | Fully Connected       | input 120, output 84                          |
  | RELU                    | input 84, output 84                           |
  | Dropout                | Keep probability 0.4
  | Fully Connected       | input 84, output 34                           |
  | softmax                | input 84, output 34                           |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.



##### Epochs : 52
  ---
  Intialialy I used upto 100 epochs to get the idea where the model gets to saturation or worse where does the validation loss start increasing again due to overfitting.

  IMAGE

  As you can see, validation is highest at around 50 epochs after which overfitting starts
  significantly.

##### Learning Rate : 0.001
  ---

  fLearning rate of 0.001 seems to be ideal

  I tried learning rates for 0.005 and 0.01, graph seems to be fluctuating a lot. Basically this translates to overshooting. So, to be on the safe side I chose a lower learning rate compromising on the convergence time.

##### Dropout : 0.4
  ---

  The overfitting in model was overwhelming, so some kind of regularisation was definitely needed. I used two dropout layer with keep probability of 40%, each after a dense layer. Had a slight, 1-2% increase effect on overall validation accuracy.


##### Batch size : 128


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

  My final model results were:
  * training set accuracy of 99.7%
  * validation set accuracy of 96%
  * test set accuracy of 93.5%

  If a well known architecture was chosen:
  * What architecture was chosen? <br>
  LeNet with dropouts
  * Why did you believe it would be relevant to the traffic sign application? <br>

  LeNet was designed keeping digit recognition in mind. So, the complexity of that MNIST data is somewhat similar to that of German traffic signals. Significant number of traffic signs can be generated by basic geometrical combination of digits. Therefore increasing model complexity might not have lead to a better accuracy.

  To avoid overfitting, I introduced two dropout layers, each after fully connected layers with keep probability of 40%.

  As thought, It gave 96% accuracy on validation set but having said that there is always room for improvement.


  * How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? <br>

  I ploted accuracy and loss Vs epochs on training and validation set to get a peak inside the training process.

  ![alt text][image2]




  Meanwhile Test set was never seen. Test accuracy was only calculated when I was done with all the preprocessing, hyperparameter optimization and model tweaking.

  Test Accuracy: 93.5%


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

  Here are five German traffic signs that I found on the web:

  ![alt text][image4] ![alt text][image5] ![alt text][image6] 
  ![alt text][image7] ![alt text][image8]

  The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

  Here are the results of the prediction:

  | Image                    |     Prediction                                | 
  |:---------------------:|:---------------------------------------------:| 
  | Stop Sign              | Stop sign                                       | 
  | Caution                 | Caution                                         |
  | Rightofway                    | Rightofway                                            |
  | Bumpy Road                  | Bicycle Crossing                                     |
  | Road work            | Road work                                  |


  The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

  The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

  Model strongly sure about its prediction and is correct too

  | Probability             |     Prediction                                | 
  |:---------------------:|:---------------------------------------------:| 
  | 0.996016800403595 | Stop |
  | 0.003980719018727541 | No entry |
  | 1.8144268096875749e-06 | Keep right |
  | 5.385537065194512e-07 | Speed limit (30km/h) |
  | 1.0166056085836317e-07 | Turn left ahead |


  Model strongly sure about its prediction and is correct too
  | Probability             |     Prediction                                | 
  |:---------------------:|:---------------------------------------------:| 
  | 0.995053768157959 | General caution |
  | 0.0049446760676801205 | Pedestrians |
  | 1.4830936834187014e-06 | Right-of-way at the next intersection |
  | 2.2216611883507653e-10 | Traffic signals |
  | 1.5970093303341315e-11 | Road narrows on the right |

  Model strongly sure about its prediction and is correct too

  | Probability             |     Prediction                                | 
  |:---------------------:|:---------------------------------------------:| 
  | 0.9988999366760254 | Right-of-way at the next intersection |
  | 0.0009981442708522081 | Beware of ice/snow |
  | 0.00010194857168244198 | Double curve |
  | 1.1880657048435328e-12 | Priority road |
  | 6.706073007014049e-14 | Pedestrians |

  Model strongly sure about its prediction and is correct too

  | Probability             |     Prediction                                | 
  |:---------------------:|:---------------------------------------------:| 
  | 0.9829909801483154 | Bicycles crossing |
  | 0.01587645150721073 | Children crossing |
  | 0.0010167069267481565 | Bumpy road |
  | 0.00011338697368046269 | Slippery road |
  | 1.5058408280310687e-06 | Dangerous curve to the right |

  Model strongly sure about its prediction and is correct too

  | Probability             |     Prediction                                | 
  |:---------------------:|:---------------------------------------------:| 
  | 1.0 | Road work |
  | 1.095911217135169e-13 | Dangerous curve to the right |
  | 7.960785791693865e-15 | Beware of ice/snow |
  | 7.003637926021795e-16 | Bumpy road |
  | 4.325957139096506e-16 | Go straight or right |
