# COURSERA-ML_with_TF_certif
MOOC to prepare TF certification

## Part 1 - Introduction to TF for AI

### Week 1

Traditional Programming: Rules + Data => Answers
Machine Learning: Data + Answers => Rules

Dense Layer: A layer of connected neurons

Loss function measures how good the current ‘guess’ is

Optimizer generates a new and improved guess

Convergence is the process of getting very close to the correct answer

The model.fit t trains the neural network to fit one set of values to another

### Week 2

 Relu: It only returns x if x is greater than zero
 Softmax takes a set of values, and effectively picks the biggest one

 split data into training and test sets To test a network with previously unseen data

### Week 3: CNN

Convolution: A technique to isolate features in images
Convolutions improve image recognition: They isolate features in images
Applying Convolutions on top of our Deep neural network will make training: It depends on many factors. It might make your training faster or slower, and a poorly designed Convolutional layer may even be less efficient than a plain DNN!
'overfitting' occurs when the network learns the data from the training set really well, but it's too specialised to only that data, and as a result is less effective at seeing other data.

[Image Filtering](https://lodev.org/cgtutor/filtering.html)

[GitHub Coursera classe](https://github.com/lmoroney/dlaicourse)

[CNN with Andrew NG](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF)

TODO >> Exercise 3 - Improve MNIST with convolutions

### Week 4: apply convolutional neural networks to much bigger and more complex images

ImageGenerator
sigmoid is great for binary classification
[Cross Entropy Loss](https://gombru.github.io/2018/05/23/cross_entropy_loss/)

[RMSProp & Momentum](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

Image Generator labels images: It’s based on the directory the image is contained in

Image Generator used rescale method to normalize the image

The target_size parameter specifies the training size for the images on the training generator


TODO >> Exercise 4 - Handling complex images

## Part 2 - Convolutional Neural Networks

### Week 1:

If my Image is sized 150x150, and I pass a 3x3 Convolution over it, the size of the resulting image is 148x148
If my data is sized 150x150, and I use Pooling of size 2x2, the size of the resulting image is 75x75
If I want to view the history of my training,, I create a variable ‘history’ and assign it to the return of model.fit or model.fit_generator
The model.layers API allows you to inspect the impact of convolutions on the images
The validation accuracy is based on images that the model hasn't been trained with, and thus a better indicator of how the model will perform with new images.

The flow_from_directory give you on the ImageGenerator : the ability to easily load images for training, the ability to pick the size of training images and the ability to automatically label images based on their directory name

Overfitting more likely to occur on smaller datasets because there's less likelihood of all possible features being encountered in the training process.


## Part 4 - Sequence & Time Series prediction

### Week 1

Use of statistical method


### Week 2

Use of DNN

Sequence bias is when the order of things can impact the selection of things. 



### Week 3

For this week, we've built on your DNNs with RNNs and LSTMs

LearningRateScheduler

Huber loss function : a loss function used in robust regression, that is less sensitive to outliers in data than the squared error loss.

Clears out all temporary variables that TF might have from previous sessions => tf.keras.backend.clear_session

Defines the dimension index at which you will expand the shape of the tensor => tf.expand_dims

Allows you to execute arbitrary code while training => Lambda layer


### Week 4


Use of Conv1D