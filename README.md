# COURSERA-ML_with_TF_certif
My notes / MOOC to prepare TF certification

## Part 1 - Introduction to TF for AI

### Week 1 (function)

- Traditional Programming: Rules + Data => Answers
- Machine Learning: Data + Answers => Rules
- Dense Layer: A layer of connected neurons 
- Loss function measures how good the current ‘guess’ is
- Optimizer generates a new and improved guess
- Convergence is the process of getting very close to the correct answer
- The model.fit trains the neural network to fit one set of values to another

### Week 2 (fashion MNIST)

- Relu: It only returns x if x is greater than zero
- Softmax takes a set of values, and effectively picks the biggest one
- Split data into training and test sets To test a network with previously unseen data


`Load from tf.keras.datasets
Normalizing images
Activation function: relu, softmax
Optimizers: adam
Sparse_categorical_crossentropy: integer form
Categorical_crossentropy: 1-hot encoding
flatten
CallBack on_epoch_end
History.epoch, history.history['acc']`

### Week 3: CNN (fachion MNSIT)

- Convolution: A technique to isolate features in images
- Convolutions improve image recognition: They isolate features in images
- Applying Convolutions on top of our Deep neural network will make training: It depends on many factors. It might make your training faster or slower, and a poorly designed Convolutional layer may even be less efficient than a plain DNN!
- 'overfitting' occurs when the network learns the data from the training set really well, but it's too specialised to only that data, and as a result is less effective at seeing other data.

Some links:
- [Image Filtering](https://lodev.org/cgtutor/filtering.html)
- [GitHub Coursera classe](https://github.com/lmoroney/dlaicourse)
- [CNN with Andrew NG](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF)

`Conv2D, MaxPooling2D`

Demos:
- Visualize Conv & Pooling => cf notebook 1
- How conv works => cf notebook 2

**TODO >> Exercise 3 - Improve MNIST with convolutions**

### Week 4: apply convolutional neural networks to much bigger and more complex images (horse vs humans)

- ImageGenerator
  * Image Generator labels images: It’s based on the directory the image is contained in
  * Image Generator used rescale method to normalize the image
  * The target_size parameter specifies the training size for the images on the training generator
- Sigmoid is great for binary classification

Some links:
- [Cross Entropy Loss](https://gombru.github.io/2018/05/23/cross_entropy_loss/)
- [RMSProp & Momentum](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

`ZipFile => Gestion des répertoires
Binary-croosentropy

datagen = ImageDataGenerator(rescale=1/255)
datagen.flow_from_directory src_dir, target_size(XX,XX), batch_size=256, class_mode="binary")`

Demo:
- Visualize intermediaite representation

Predict of one image => np.expand !

**TODO >> Exercise 4 - Handling complex images**

## Part 2 - Convolutional Neural Networks

### Week 1: (cats & dogs)

- If my Image is sized 150x150, and I pass a 3x3 Convolution over it, the size of the resulting image is 148x148
- If my data is sized 150x150, and I use Pooling of size 2x2, the size of the resulting image is 75x75
- If I want to view the history of my training,, I create a variable ‘history’ and assign it to the return of model.fit or model.fit_generator
- The model.layers API allows you to inspect the impact of convolutions on the images
- The validation accuracy is based on images that the model hasn't been trained with, and thus a better indicator of how the model will perform with new images.
- The flow_from_directory give you on the ImageGenerator : the ability to easily load images for training, the ability to pick the size of training images and the ability to automatically label images based on their directory name
- Overfitting more likely to occur on smaller datasets because there's less likelihood of all possible features being encountered in the training process.

`Plot Acc & Loss`

**TODO >> Ungraded Exercice Let's start building a classifier using the full Cats v Dogs dataset of 25k images.**

### Week 2: Image Augmentation (cats & dogs)

- ImageDataGenerator:
  * rotation_range is a value in degrees (0–180), a range within which to randomly rotate pictures.
  * width_shift and height_shift are ranges (as a fraction of total width or height) within which to randomly translate pictures vertically or horizontally.
  * shear_range is for randomly applying shearing transformations.
  * zoom_range is for randomly zooming inside pictures.
  * horizontal_flip is for randomly flipping half of the images horizontally. This is relevant when there are no assumptions of horizontal assymmetry (e.g. real-world pictures).
  * fill_mode is the strategy used for filling in newly created pixels, which can appear after a rotation or a width/height shift. => It attempts to recreate lost information after a transformation like a shear

- The image augmentation introduces a random element to the training images but if the validation set doesn't have the same randomness, then its results can fluctuate like this. 
- All augmentation is done in-memory
- When training with augmentation, you noticed that the training is a little slower because the image processing takes cycles.
- Add Augmentation to it, and experiment with different parameters to avoid overfitting. This will likely take a lot of time -- as it requires using the full dataset along with augmentation code to edit the data on-the-fly.
  * 1) All the original images are just transformed (i.e. rotation, zooming, etc.) every epoch and then used for training, and 
  * 2) [Therefore] the number of images in each epoch is equal to the number of original images you have.

`ÌmageDataGenerator`

### Week 3: Transfer Learning (cats & dogs / horses vs humans)

What we learn?
Transfer Learning: you can take an existing model, freeze many of its layers to prevent them being retrained, and effectively 'remember' the convolutions it was trained on to fit images. You then added your own DNN underneath this so that you could retrain on your images using the convolutions from the other model. 
You learned about regularization using dropouts to make your network more efficient in preventing over-specialization and this overfitting.

Good Intro for def TL

Video 1 => good schema

BatchNormalization & TL:
- Many models contain tf.keras.layers.BatchNormalization layers. This layer is a special case and precautions should be taken in the context of fine-tuning, as shown later in this tutorial.
- When you set layer.trainable = False, the BatchNormalization layer will run in inference mode, and will not update its mean and variance statistics.
- When you unfreeze a model that contains BatchNormalization layers in order to do fine-tuning, you should keep the BatchNormalization layers in inference mode by passing training = False when calling the base model. Otherwise, the updates applied to the non-trainable weights will destroy what the model has learned. 

TL Principles:
- We saw how to take the layers from an existing model, and make them so that they don't get retrained -- i.e. we freeze (or lock) the already learned convolutions into your model. 
- After that, we have to add our own DNN at the bottom of these, which we can retrain to our data.

Dropout:
- The idea behind Dropouts is that they remove a random number of neurons in your neural network. 
- This works very well for two reasons: 
 - The first is that neighboring neurons often end up with similar weights, which can lead to overfitting, so dropping some out at random can remove this. 
 - The second is that often a neuron can over-weight the input from a neuron in the previous layer, and can over specialize as a result. Thus, dropping out can break the neural network out of this potential bad habit! 

`Ìnception
include_top
load_wieghts
dropout`

**TODO >> Ungraded Exercice**

### Week 4: Multiclass Classifications (Rock Paper Scissors)

[Rock Paper Scissors Dataset](http://www.laurencemoroney.com/rock-paper-scissors-dataset/)

## Part 3 - Natural Language Processing in TensorFlow

### Week 1: Sentiment in text (BBC news)

1-how to tokenize the words and sentences, building up a dictionary of all the words to make a corpus

Word based encodings :
- if encoding character in ASCII =>  semantics of the word aren't encoded in the letters => Silent != Listen
- we have a value per word, and the value is the same for the same word every time

`keras.preprocessing.text.Tokenizer(num_words)`
- method used to tokenize a list of sentences: `fit_on_texts(sentences)`
- `word_index`
- method used to encode a list of sentences to use those tokens: `text_to_sequences(sentences)`
- Out Of Vocabulary: `keras.preprocessing.text.Tokenizer(num_words, oov_token="<OOV>")`

Padding: `tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post', truncating='post', maxlen=5)`

### Week 2 (IMDB)

Embedding, with the idea being that words and associated words are clustered as vectors in a multi-dimensional space. 
= It is the number of dimensions for the vector representing the word encoding

TensorFlow Data Services or TFDS contains many data sets and lots of different categories

http://projector.tensorflow.org/

When using IMDB Sub Words dataset, our results in classification were poor. Why? Sequence becomes much more important when dealing with subwords, but we’re ignoring word positions

`tfds
binary_crossentropy
embeddings`

Demo:
- vec+meta = visu embeddings

### Week 3 (IMDB subwords / Sarcasm / IMDB) 

output shape of a bidirectional LSTM layer with 64 units is (None, 128)

`bidirectional (LSTM)
multilayer
conv1D - globalAveragePool1D`

### Week 4 (Sonnets)

Generate shakespeare text

`bidirectional (LSTM)
dropout`

## Part 4 - Sequence & Time Series prediction

### Week 1: Use of statistical method


### Week 2: Use of DNN

- Sequence bias is when the order of things can impact the selection of things. 

### Week 3: DNNs with RNNs and LSTMs

- LearningRateScheduler
- Huber loss function : a loss function used in robust regression, that is less sensitive to outliers in data than the squared error loss.
- Clears out all temporary variables that TF might have from previous sessions => tf.keras.backend.clear_session
- Defines the dimension index at which you will expand the shape of the tensor => tf.expand_dims
- Allows you to execute arbitrary code while training => Lambda layer

### Week 4: Use of Conv1D

## More and more....
[Colab Files](https://neptune.ai/blog/google-colab-dealing-with-files)
Save your model “efficiently”:
1. Use EarlyStopping() keras callback (with restore_best_weights=True) to stop training before overfitting while reserving best weights so far.
2. Use [ModelCheckpoint()](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint) keras callback (with save_best_only=True) to save a copy of your model whenever it gets better.
3. Use include_optimizer=False option in your keras.models.save_model (or model.save) statement, to reduce the size of your model. 
