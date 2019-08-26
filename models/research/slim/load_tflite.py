import time

import cv2
import numpy
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

#data = input_data.read_data_sets("mnist_data", one_hot=True)

#test = data.test

# load TFlite model ===========================================
interpreter = tf.contrib.lite.Interpreter(model_path="./mobilenet_qt_v1_224.tflite")
interpreter.allocate_tensors()

# Get input and output tensor ==================================
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# test model
input_shape = input_details[0]["shape"]
print(input_shape)

test_imgs = test.images
test_labels = test.labels

correct_prediction = 0.
mean_time = 0.
for i in range(len(test_imgs)):
    temp = numpy.reshape(cv2.normalize(test_imgs[i], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U),
                         newshape=[1, 784])
    interpreter.set_tensor(input_details[0]["index"], temp)
    interpreter.invoke()
    start = time.time()
    output = interpreter.get_tensor(output_details[0]["index"])
    mean_time += time.time() - start
    correct_prediction += np.equal(numpy.argmax(output), numpy.argmax(test_labels[i]))

correct_prediction = correct_prediction/float(len(test_imgs))
print("time_elapse: ", mean_time/float(len(test_imgs)))
print(correct_prediction)
