from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

#data set link
training = "iris_training.csv"
test = "iris_test.csv"

#load data
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=training, target_dtype = np.int, features_dtype = np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=test, target_dtype = np.int, features_dtype = np.float32)

#make columns for each feature (thers 4 of them in this iris data set)
#specifies that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

#construct DNNClassifier. 3 layers; 10, 20, 10 nodes each
#feature_columns: mentioned above
#hidden unites: specifies 3 layers of 10, 20, 10 nodes respectively
#n_classes: the three iris species in the data set
#model_dir: where tensorflow will say checkpoint data
classifier = tf.contrib.learn.DNNClassifier(feature_columns = feature_columns, hidden_units = [10,20,10], n_classes = 3, model_dir = "/tmp/iris_model")

#fit the training model
#use monitor if you want to track the progress directly
classifier.fit(x=training_set.data, y=training_set.target, steps = 2000)

#acc eval
accuracy = classifier.evaluate(x=test_set.data, y=test_set.target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy))


#if a new input is found, it is to be input like this:
new_sample = np.array([[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype = float)
# this inputs two new samples indicated in a nested array:
# [6.4, 3.2, 4.5, 1.5] and [5.8, 3.1, 5.0, 1.7] will be classified
y = list(classifier.predict(new_sample, as_iterable = True))
print('Predictions for new samples: {}'.format(str(y)))
