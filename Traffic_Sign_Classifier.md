
# Self-Driving Car Engineer Nanodegree

## Deep Learning

## Project: Build a Traffic Sign Recognition Classifier

In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 

> **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 

In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.

The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.


>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

The code below was for downloading the zip file because it made it easier to work in Google colab with this.


```python
# Load pickled data
import pickle
import os
from urllib.request import urlretrieve

def download(url, file):
    """
    Download file from <url>
    :param url: URL to file
    :param file: Local file path
    """
    if not os.path.isfile(file):
        print('Downloading ' + file + '...')
        urlretrieve(url, file)
        print('Download Finished')

download('https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip', 'download.zip')

```

---
## Step 0: Load The Data


```python
import zipfile
zip_ref = zipfile.ZipFile('download.zip', 'r')
zip_ref.extractall(".")
zip_ref.close()
import pickle

training_file = "train.p"
validation_file="valid.p"
testing_file = "test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
```

---

## Step 1: Dataset Summary & Exploration

The pickled data is a dictionary with 4 key/value pairs:

- `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
- `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
- `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
- `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**

Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 

### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas


```python
### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results
import numpy as np
# TODO: Number of training examples
n_train = np.shape(y_train)[0]

# TODO: Number of validation examples
n_validation = np.shape(y_valid)[0]

# TODO: Number of testing examples.
n_test = np.shape(y_test)[0]

# TODO: What's the shape of an traffic sign image?
image_shape = np.shape(X_train)[1:]

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))
print(np.shape(y_train))
print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Number of validation examples =", n_validation)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
```

    (34799,)
    Number of training examples = 34799
    Number of testing examples = 12630
    Number of validation examples = 4410
    Image data shape = (32, 32, 3)
    Number of classes = 43
    

### Include an exploratory visualization of the dataset

Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc. 

The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.

**NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections. It can be interesting to look at the distribution of classes in the training, validation and test set. Is the distribution the same? Are there more examples of some classes than others?


```python
### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
%matplotlib inline

import csv
with open('signnames.csv') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    signs = []
    for row_id,row in enumerate(csv_reader):
        if row_id != 0:
            signs.append(row[1])
            
print(signs)
```

    ['Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)', 'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)', 'No passing', 'No passing for vehicles over 3.5 metric tons', 'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop', 'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 'No entry', 'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right', 'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right', 'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing', 'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory', 'End of no passing', 'End of no passing by vehicles over 3.5 metric tons']
    


```python
import pandas as pd

def plot_histogram(y_train):
    classes = pd.DataFrame()
    classes['label'] = y_train
    ax = classes['label'].value_counts().plot(kind='barh', figsize = (15,15), title='No. of images per class')
    ax.set_yticklabels(list(map(lambda x: signs[x], classes['label'].value_counts().index.tolist()))) 
    for i, v in enumerate(classes['label'].value_counts()):
        ax.text(v + 10, i - 0.25, str(v), color='blue')
plot_histogram(y_train)
```


![png](output_11_0.png)


----

## Step 2: Design and Test a Model Architecture

Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play! 

With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission. 

There are various aspects to consider when thinking about this problem:

- Neural network architecture (is the network over or underfitting?)
- Play around preprocessing techniques (normalization, rgb to grayscale, etc)
- Number of examples per label (some have more than others).
- Generate fake data.

Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

### Pre-process the Data Set (normalization, grayscale, etc.)

Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data and can be used in this project. 

Other pre-processing steps are optional. You can try different techniques to see if it improves performance. 

Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.


```python
### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)

def grey(y):
  y = (np.mean(y,-1))
  y = np.reshape(y, y.shape + (1,))
  return y

x_train = []
x_valid = []
x_test = []
for i in range(n_train):
  x_train.append(grey(X_train[i]))
for i in range(n_validation):
  x_valid.append(grey(X_valid[i]))
for i in range(n_test):
  x_test.append(grey(X_test[i]))
  
X_train = np.array(x_train)
X_valid = np.array(x_valid)
X_test = np.array(x_test)

def normalize(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    a = 0.1
    b = 0.9
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )
  

#print(X_train[0])
X_train = normalize(X_train)
X_test = normalize(X_test)
X_valid = normalize(X_valid)
#print(X_train[0])
```

### Model Architecture


```python
### Define your architecture here.
### Feel free to use as many code cells as needed.
EPOCHS = 100
BATCH_SIZE = 128
from tensorflow.contrib.layers import flatten

def LeNet(x):    
    # Hyperparameters
    mu = 0
    sigma = 0.1

    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    print(conv1)
    # SOLUTION: Activation.
    conv1_relu = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1_pool = tf.nn.max_pool(conv1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2  = tf.nn.conv2d(conv1_pool, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

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
    fc1 = tf.nn.dropout(fc1 , keep_prob)
    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b

    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2 , keep_prob)
    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits
```

### Train, Validate and Test the Model

A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.


```python
import tensorflow as tf
### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, 43)
rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
```

    Tensor("add:0", shape=(?, 28, 28, 6), dtype=float32)
    


```python
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y , keep_prob:1})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
```


```python
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
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y , keep_prob:0.8 })
            
        validation_accuracy = evaluate(X_valid, y_valid)
        train_accuracy = evaluate(X_train, y_train)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print("Training Accuracy = {:.3f}".format(train_accuracy))
        
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")
```

    Training...
    
    EPOCH 1 ...
    Validation Accuracy = 0.539
    Training Accuracy = 0.594
    
    EPOCH 2 ...
    Validation Accuracy = 0.792
    Training Accuracy = 0.832
    
    EPOCH 3 ...
    Validation Accuracy = 0.842
    Training Accuracy = 0.889
    
    EPOCH 4 ...
    Validation Accuracy = 0.858
    Training Accuracy = 0.918
    
    EPOCH 5 ...
    Validation Accuracy = 0.892
    Training Accuracy = 0.944
    
    EPOCH 6 ...
    Validation Accuracy = 0.901
    Training Accuracy = 0.951
    
    EPOCH 7 ...
    Validation Accuracy = 0.907
    Training Accuracy = 0.960
    
    EPOCH 8 ...
    Validation Accuracy = 0.906
    Training Accuracy = 0.965
    
    EPOCH 9 ...
    Validation Accuracy = 0.915
    Training Accuracy = 0.976
    
    EPOCH 10 ...
    Validation Accuracy = 0.912
    Training Accuracy = 0.978
    
    EPOCH 11 ...
    Validation Accuracy = 0.924
    Training Accuracy = 0.981
    
    EPOCH 12 ...
    Validation Accuracy = 0.928
    Training Accuracy = 0.984
    
    EPOCH 13 ...
    Validation Accuracy = 0.921
    Training Accuracy = 0.986
    
    EPOCH 14 ...
    Validation Accuracy = 0.926
    Training Accuracy = 0.983
    
    EPOCH 15 ...
    Validation Accuracy = 0.930
    Training Accuracy = 0.990
    
    EPOCH 16 ...
    Validation Accuracy = 0.927
    Training Accuracy = 0.990
    
    EPOCH 17 ...
    Validation Accuracy = 0.935
    Training Accuracy = 0.990
    
    EPOCH 18 ...
    Validation Accuracy = 0.934
    Training Accuracy = 0.992
    
    EPOCH 19 ...
    Validation Accuracy = 0.943
    Training Accuracy = 0.993
    
    EPOCH 20 ...
    Validation Accuracy = 0.937
    Training Accuracy = 0.993
    
    EPOCH 21 ...
    Validation Accuracy = 0.939
    Training Accuracy = 0.992
    
    EPOCH 22 ...
    Validation Accuracy = 0.941
    Training Accuracy = 0.994
    
    EPOCH 23 ...
    Validation Accuracy = 0.942
    Training Accuracy = 0.995
    
    EPOCH 24 ...
    Validation Accuracy = 0.941
    Training Accuracy = 0.995
    
    EPOCH 25 ...
    Validation Accuracy = 0.935
    Training Accuracy = 0.994
    
    EPOCH 26 ...
    Validation Accuracy = 0.936
    Training Accuracy = 0.996
    
    EPOCH 27 ...
    Validation Accuracy = 0.948
    Training Accuracy = 0.997
    
    EPOCH 28 ...
    Validation Accuracy = 0.947
    Training Accuracy = 0.997
    
    EPOCH 29 ...
    Validation Accuracy = 0.947
    Training Accuracy = 0.996
    
    EPOCH 30 ...
    Validation Accuracy = 0.946
    Training Accuracy = 0.997
    
    EPOCH 31 ...
    Validation Accuracy = 0.941
    Training Accuracy = 0.997
    
    EPOCH 32 ...
    Validation Accuracy = 0.941
    Training Accuracy = 0.998
    
    EPOCH 33 ...
    Validation Accuracy = 0.943
    Training Accuracy = 0.998
    
    EPOCH 34 ...
    Validation Accuracy = 0.939
    Training Accuracy = 0.998
    
    EPOCH 35 ...
    Validation Accuracy = 0.943
    Training Accuracy = 0.998
    
    EPOCH 36 ...
    Validation Accuracy = 0.942
    Training Accuracy = 0.996
    
    EPOCH 37 ...
    Validation Accuracy = 0.941
    Training Accuracy = 0.998
    
    EPOCH 38 ...
    Validation Accuracy = 0.945
    Training Accuracy = 0.999
    
    EPOCH 39 ...
    Validation Accuracy = 0.950
    Training Accuracy = 0.998
    
    EPOCH 40 ...
    Validation Accuracy = 0.944
    Training Accuracy = 0.998
    
    EPOCH 41 ...
    Validation Accuracy = 0.945
    Training Accuracy = 0.999
    
    EPOCH 42 ...
    Validation Accuracy = 0.953
    Training Accuracy = 0.999
    
    EPOCH 43 ...
    Validation Accuracy = 0.946
    Training Accuracy = 0.999
    
    EPOCH 44 ...
    Validation Accuracy = 0.956
    Training Accuracy = 0.999
    
    EPOCH 45 ...
    Validation Accuracy = 0.943
    Training Accuracy = 0.999
    
    EPOCH 46 ...
    Validation Accuracy = 0.949
    Training Accuracy = 0.999
    
    EPOCH 47 ...
    Validation Accuracy = 0.948
    Training Accuracy = 0.999
    
    EPOCH 48 ...
    Validation Accuracy = 0.944
    Training Accuracy = 0.999
    
    EPOCH 49 ...
    Validation Accuracy = 0.951
    Training Accuracy = 0.999
    
    EPOCH 50 ...
    Validation Accuracy = 0.941
    Training Accuracy = 0.999
    
    EPOCH 51 ...
    Validation Accuracy = 0.951
    Training Accuracy = 0.999
    
    EPOCH 52 ...
    Validation Accuracy = 0.951
    Training Accuracy = 0.999
    
    EPOCH 53 ...
    Validation Accuracy = 0.954
    Training Accuracy = 0.999
    
    EPOCH 54 ...
    Validation Accuracy = 0.957
    Training Accuracy = 1.000
    
    EPOCH 55 ...
    Validation Accuracy = 0.952
    Training Accuracy = 1.000
    
    EPOCH 56 ...
    Validation Accuracy = 0.952
    Training Accuracy = 0.999
    
    EPOCH 57 ...
    Validation Accuracy = 0.934
    Training Accuracy = 0.997
    
    EPOCH 58 ...
    Validation Accuracy = 0.956
    Training Accuracy = 1.000
    
    EPOCH 59 ...
    Validation Accuracy = 0.953
    Training Accuracy = 1.000
    
    EPOCH 60 ...
    Validation Accuracy = 0.954
    Training Accuracy = 1.000
    
    EPOCH 61 ...
    Validation Accuracy = 0.955
    Training Accuracy = 1.000
    
    EPOCH 62 ...
    Validation Accuracy = 0.951
    Training Accuracy = 1.000
    
    EPOCH 63 ...
    Validation Accuracy = 0.953
    Training Accuracy = 0.999
    
    EPOCH 64 ...
    Validation Accuracy = 0.949
    Training Accuracy = 0.999
    
    EPOCH 65 ...
    Validation Accuracy = 0.954
    Training Accuracy = 1.000
    
    EPOCH 66 ...
    Validation Accuracy = 0.954
    Training Accuracy = 1.000
    
    EPOCH 67 ...
    Validation Accuracy = 0.954
    Training Accuracy = 1.000
    
    EPOCH 68 ...
    Validation Accuracy = 0.949
    Training Accuracy = 1.000
    
    EPOCH 69 ...
    Validation Accuracy = 0.954
    Training Accuracy = 1.000
    
    EPOCH 70 ...
    Validation Accuracy = 0.947
    Training Accuracy = 1.000
    
    EPOCH 71 ...
    Validation Accuracy = 0.951
    Training Accuracy = 0.999
    
    EPOCH 72 ...
    Validation Accuracy = 0.946
    Training Accuracy = 0.999
    
    EPOCH 73 ...
    Validation Accuracy = 0.956
    Training Accuracy = 0.999
    
    EPOCH 74 ...
    Validation Accuracy = 0.956
    Training Accuracy = 1.000
    
    EPOCH 75 ...
    Validation Accuracy = 0.957
    Training Accuracy = 1.000
    
    EPOCH 76 ...
    Validation Accuracy = 0.956
    Training Accuracy = 1.000
    
    EPOCH 77 ...
    Validation Accuracy = 0.949
    Training Accuracy = 0.999
    
    EPOCH 78 ...
    Validation Accuracy = 0.954
    Training Accuracy = 1.000
    
    EPOCH 79 ...
    Validation Accuracy = 0.954
    Training Accuracy = 1.000
    
    EPOCH 80 ...
    Validation Accuracy = 0.955
    Training Accuracy = 1.000
    
    EPOCH 81 ...
    Validation Accuracy = 0.957
    Training Accuracy = 1.000
    
    EPOCH 82 ...
    Validation Accuracy = 0.955
    Training Accuracy = 1.000
    
    EPOCH 83 ...
    Validation Accuracy = 0.956
    Training Accuracy = 1.000
    
    EPOCH 84 ...
    Validation Accuracy = 0.956
    Training Accuracy = 1.000
    
    EPOCH 85 ...
    Validation Accuracy = 0.956
    Training Accuracy = 1.000
    
    EPOCH 86 ...
    Validation Accuracy = 0.956
    Training Accuracy = 1.000
    
    EPOCH 87 ...
    Validation Accuracy = 0.954
    Training Accuracy = 1.000
    
    EPOCH 88 ...
    Validation Accuracy = 0.952
    Training Accuracy = 1.000
    
    EPOCH 89 ...
    Validation Accuracy = 0.946
    Training Accuracy = 1.000
    
    EPOCH 90 ...
    Validation Accuracy = 0.954
    Training Accuracy = 1.000
    
    EPOCH 91 ...
    Validation Accuracy = 0.952
    Training Accuracy = 1.000
    
    EPOCH 92 ...
    Validation Accuracy = 0.952
    Training Accuracy = 1.000
    
    EPOCH 93 ...
    Validation Accuracy = 0.958
    Training Accuracy = 1.000
    
    EPOCH 94 ...
    Validation Accuracy = 0.950
    Training Accuracy = 1.000
    
    EPOCH 95 ...
    Validation Accuracy = 0.957
    Training Accuracy = 1.000
    
    EPOCH 96 ...
    Validation Accuracy = 0.954
    Training Accuracy = 1.000
    
    EPOCH 97 ...
    Validation Accuracy = 0.951
    Training Accuracy = 1.000
    
    EPOCH 98 ...
    Validation Accuracy = 0.951
    Training Accuracy = 1.000
    
    EPOCH 99 ...
    Validation Accuracy = 0.955
    Training Accuracy = 1.000
    
    EPOCH 100 ...
    Validation Accuracy = 0.954
    Training Accuracy = 1.000
    
    Model saved
    

The below cell is used for printing the test accuracy of the model which is more than the specified 93%.


```python
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
```

    Test Accuracy = 0.937
    

---

## Step 3: Test a Model on New Images

To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.

You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

### Load and Output the Images


```python
### Load the images and plot them here.
### Feel free to use as many code cells as needed.
import matplotlib.image as mpimg
for i in range(1,6):
    image = mpimg.imread(str(i) + '.jpg')
    plt.figure(figsize=(1,1))
    plt.imshow(image)
```


![png](output_27_0.png)



![png](output_27_1.png)



![png](output_27_2.png)



![png](output_27_3.png)



![png](output_27_4.png)


### Predict the Sign Type for Each Image


```python
### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.
predicted_classes = []
a = [1 ,1 ,1 ,1 ,1 ]
for i in range(6):
    image = mpimg.imread(str(i) + '.jpg')
    img = grey(image)
    img = normalize(img)
    a[i-1] = img
stack_ = np.stack((a[0] , a[1] , a[2] , a[3] , a[4]))
print(np.shape(stack_))
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    res1 = sess.run([logits,tf.argmax(logits, 1)] , feed_dict={x: stack_, keep_prob:1 })
    print(res1)
```

    (5, 32, 32, 1)
    [array([[ -47.08484  ,  -20.229351 ,  -56.450863 ,  -68.4235   ,
             -73.94116  ,  -28.479431 ,  -51.010242 ,  -55.51414  ,
             -82.48129  ,  -69.28644  ,  -19.394423 ,   -8.809617 ,
             -47.898037 ,  -54.207493 ,  -34.527657 , -110.871155 ,
             -86.72843  ,  -36.246773 ,  -22.559792 ,  -18.404615 ,
              -9.997904 ,    8.543758 ,  -15.906614 ,    9.114038 ,
               9.538073 ,   34.30154  ,  -26.405706 ,  -38.767616 ,
             -33.11634  ,    8.532858 ,   14.877828 ,   11.269914 ,
             -96.14135  ,  -24.37614  ,  -17.673328 ,  -23.893972 ,
             -67.09335  ,   -2.8280761,  -11.842466 ,  -33.80124  ,
             -45.130173 ,  -50.432617 ,  -61.361732 ],
           [ -85.60871  ,  -61.171494 ,  -55.27032  ,  -28.949543 ,
            -153.31331  ,  -44.19399  ,  -35.087715 ,  -38.731846 ,
             -58.719532 ,  -26.787186 ,  -52.32281  ,   19.787977 ,
             -23.942543 ,  -55.340473 , -129.062    ,  -60.275238 ,
             -33.29817  ,  -57.103397 ,  -40.049522 ,  -17.522243 ,
              -0.2095573,  -43.81606  ,  -80.61536  ,   -5.599496 ,
             -10.130051 ,  -15.093129 ,   -9.630046 ,   -9.590603 ,
              22.462877 ,  -18.42386  ,   19.366003 ,  -59.633842 ,
             -31.94123  ,  -59.16955  ,  -22.735872 ,  -24.22779  ,
             -42.748177 ,  -49.74894  ,  -21.130861 ,  -84.68413  ,
             -42.82409  ,   -4.559646 ,  -15.020247 ],
           [-105.849884 ,  -36.322    ,  -53.729645 ,  -40.973103 ,
            -124.44571  ,  -45.616894 ,  -62.549267 ,  -35.470016 ,
             -97.36188  ,  -52.338036 ,  -82.05181  ,   43.964928 ,
              -9.589034 ,  -65.29514  , -131.35896  , -109.48215  ,
             -27.175627 ,  -56.00354  ,   -2.602058 ,    2.7887282,
             -65.231804 ,   13.015811 , -163.29167  ,  -42.52831  ,
             -31.59546  ,  -52.922733 ,  -39.53762  ,   15.712476 ,
             -16.378778 ,  -51.696297 ,  -15.253965 ,  -51.108997 ,
             -66.34842  ,  -13.838922 ,  -63.977833 ,  -30.944626 ,
             -62.475086 ,  -44.897522 ,  -48.8389   ,  -51.314    ,
             -20.257914 ,  -36.143265 ,  -43.923813 ],
           [ -51.31537  ,  -10.89247  ,   25.368862 ,   29.883623 ,
             -58.44862  ,   -5.9393096,  -58.6624   ,  -45.608772 ,
             -82.342445 ,  -26.760881 ,  -16.005402 ,  -46.55619  ,
             -29.709293 ,  -21.193794 ,  -33.979073 ,  -26.02318  ,
             -82.751144 ,  -73.64802  ,  -57.22178  ,  -28.276915 ,
             -41.252457 ,  -52.40079  ,  -62.810566 ,   -4.5863695,
             -55.774002 ,  -17.05085  ,  -81.055    ,  -72.10551  ,
             -17.699432 ,  -28.619907 ,  -83.167816 ,   -3.462204 ,
             -60.07947  ,  -54.657017 ,  -69.90061  ,  -46.85525  ,
             -54.81414  ,  -61.274708 ,    4.0251975,  -60.559956 ,
             -45.03707  ,  -59.62508  ,  -50.76292  ],
           [ -24.44988  ,  -44.666664 ,  -28.755287 ,  -13.207755 ,
             -59.674465 ,  -16.308521 ,  -26.936932 ,  -32.586132 ,
             -20.605448 ,  -16.37674  ,   -9.590779 ,   -0.5112717,
             -39.35515  ,  -59.206905 ,  -51.62552  ,  -37.907024 ,
              -8.288342 ,  -19.496193 ,   -7.8847337,    4.691418 ,
              16.437931 ,   -9.10867  ,  -15.118123 ,   17.492962 ,
              -7.554326 ,   -5.017102 ,  -11.020779 ,  -12.690172 ,
               5.692894 ,    3.6890438,    5.2473373,   12.01762  ,
             -34.462837 ,  -45.1636   ,  -20.704195 ,  -32.66797  ,
             -21.765392 ,  -11.679793 ,  -14.136222 ,  -44.602993 ,
             -12.577013 ,  -17.14207  ,  -34.04923  ]], dtype=float32), array([25, 28, 11,  3, 23], dtype=int64)]
    

### Analyze Performance


```python
### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.
true_classes = [25 , 28 , 11 , 3 , 31]
perf = np.sum(true_classes == res1[1])
print("Accuracy is " + str(perf*100/5) + "%.")
```

    Accuracy is 80.0%.
    

### Output Top 5 Softmax Probabilities For Each Image Found on the Web

For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 

The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.

`tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.

Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tf.nn.top_k` is used to choose the three classes with the highest probability:

```
# (5, 6) array
a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
         0.12789202],
       [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
         0.15899337],
       [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
         0.23892179],
       [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
         0.16505091],
       [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
         0.09155967]])
```

Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:

```
TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
       [ 0.28086119,  0.27569815,  0.18063401],
       [ 0.26076848,  0.23892179,  0.23664738],
       [ 0.29198961,  0.26234032,  0.16505091],
       [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
       [0, 1, 4],
       [0, 5, 1],
       [1, 3, 5],
       [1, 4, 3]], dtype=int32))
```

Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.


```python
### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.
# def softmax(L):
#     xx = np.exp(L)
#     return xx/sum(xx)*1.0

# softmaxes = []
# for arr in res1[0]:
#     #print(arr)
#     softmaxes.append(softmax(arr))
# print(softmaxes)
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    result = sess.run(tf.nn.top_k(tf.nn.softmax(res1[0]), k=5))
    print(result)
```

    TopKV2(values=array([[1.0000000e+00, 3.6676537e-09, 9.9423997e-11, 1.7593919e-11,
            1.1513488e-11],
           [8.9758235e-01, 6.1855733e-02, 4.0561959e-02, 1.2780899e-10,
            1.6494582e-12],
           [1.0000000e+00, 5.3717480e-13, 3.6221782e-14, 1.3103978e-18,
            5.9733079e-21],
           [9.8917222e-01, 1.0827699e-02, 5.8223799e-12, 3.2610995e-15,
            1.0596048e-15],
           [7.3943359e-01, 2.5745723e-01, 3.0973367e-03, 5.5487453e-06,
            3.5537919e-06]], dtype=float32), indices=array([[25, 30, 31, 24, 23],
           [28, 11, 30, 20, 41],
           [11, 27, 21, 19, 18],
           [ 3,  2, 38, 31, 23],
           [23, 20, 31, 28, 30]]))
    

### Project Writeup

Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 

> **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

---

## Step 4 (Optional): Visualize the Neural Network's State with Test Images

 This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.

 Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the [LeNet lab's](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.

For an example of what feature map outputs look like, check out NVIDIA's results in their paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.

<figure>
 <img src="visualize_cnn.png" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Your output should look something like this (above)</p> 
 </figcaption>
</figure>
 <p></p> 



```python
### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
```


Writeup:

1.I used the numpy library to calculate summary statistics of the traffic signs data set:
The size of training set is 34799
The size of test set is 12630
The shape of a traffic sign image is (32, 32, 3)
The number of unique classes/labels in the data set is 43

2. I was having a look at all the different different classes present using the module csv and also the unequal distributiom
   of different classes as Speed Limit(20km/h) had only 180 images but Speed Limit(50km/h) had 2010 images.

3. As a first step, I decided to convert the images to grayscale because in case with traffic signs, the color is unlikely to give any performance boost. As a second step, I normalize the image data between 0.1 and 0.9 as we had done in the class to better concentrate the data for quicker Gradient descent.

| Layer        | Description    |
| :------------- :|:-------------|
| Input      | 5x5	1x1 stride, 'VALID' padding, outputs 28x28x6 |
| RELU     |       |
| Max pooling | 2x2 stride, outputs 14x14x6      |
|Convolution | 5x5	1x1 stride, 'VALID' padding, outputs 10x10x16     |
| RELU |      |
| Max pooling | 2x2 stride, outputs 5x5x6      |
| Flatten| outputs 400   |
| Dropout	keep probability | 0.8     |
| Fully connected	output | 120      |
| RELU |      |
| Dropout	keep probability | 0.8     |
| Fully connected	output | 84      |
| RELU |      |
| Dropout	keep probability | 0.8     |
| Fully connected	output | 43      |

To train the model,following parameters were used:

Number of epochs = 100
Batch size = 128
Learning rate = 0.001
Optimizer - Adam algorithm (alternative of stochastic gradient descent). Optimizer uses backpropagation to update the network and minimize training loss.
Dropout = 0.8 (training only)


My final model results were:

training set accuracy of 100%
validation set accuracy of 95.4%
test set accuracy of 93.7%

The number of epochs are 100 but validation accuracy of 93% was acheived at 17th epoch itself. Data augmentation was done but even after changing of parameters the quality degraded so was removed ultimately. Dropout layers addition was the biggest upgrade and that was solely responsible for boosting the accuracy. 

5 different images found on web:
[//]: # (Image References)

[image1]: ./1.jpg
![alt text][image1]
[image2]: ./2.jpg
![alt text][image2]
[image3]: ./3.jpg
![alt text][image3]
[image4]: ./4.jpg
![alt text][image4]
[image5]: ./5.jpg
![alt text][image5]

| Image        | Prediction    |
| ------------- |:-------------:|
| Road work      | Road Work |
| Children crossing      | Children crossing |
| Right-of-way      | Right-of-way |
| Speed limit (60km/h)      | Speed limit (60km/h) |
| Wild animals crossing      | Road work |



The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%.
