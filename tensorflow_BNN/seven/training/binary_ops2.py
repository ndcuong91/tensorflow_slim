# -*- coding: utf-8 -*-
from __future__ import absolute_import
import tensorflow as tf

def _soft_tanh(x):
    return tf.where(tf.greater(tf.abs(x), 1.), x*0.0001, x)

def _soft_sigmoid(x):
    return tf.where(tf.greater(x, 1.), x*0.0001,
             tf.where(tf.less(x, 0.), x*0.0001, x))

def bin_table(x):
    y = _soft_tanh(x)
    quantized = tf.where(tf.greater_equal(y, 0), tf.fill(tf.shape(y), 1.), tf.fill(tf.shape(y), -1.))
    return y + tf.stop_gradient(quantized - y)

def binarize(W, H=1, stochastic=False):
    # [-H, H] -> -H or H
    Wb = H * bin_table(W / H)
    return Wb


def range_prelu(x, upper_limit=4., outer_slope=0.0001):
    '''Quatery ReLU (0,4)
    '''
    return tf.where(tf.logical_or(tf.greater(x, upper_limit), tf.less(x, 0.)), x*outer_slope, x)


def binary_relu(x):
    #y = tf.nn.relu(x)
    y = range_prelu(x, upper_limit=1.)
    #y = _soft_sigmoid(x)
    quantized = tf.where(tf.greater_equal(y, 0.5), tf.fill(tf.shape(y), 1.), tf.fill(tf.shape(y), 0.))
    return y + tf.stop_gradient(quantized - y)

def quatery_relu(x):
    #y = tf.nn.relu(x)
    y = range_prelu(x, upper_limit=4.)
    quantized = tf.where(tf.greater_equal(y, 3.),  tf.fill(tf.shape(y), 4.),
                  tf.where(tf.greater_equal(y, 1.5), tf.fill(tf.shape(y), 2.),
                    tf.where(tf.greater_equal(y, 0.5), tf.fill(tf.shape(y), 1.),
                      tf.fill(tf.shape(y), 0.) ) ) )
    return y + tf.stop_gradient(quantized - y)

