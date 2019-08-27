# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Model definitions for simple speech recognition.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf


def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms,
                           dct_coefficient_count):
  """Calculates common settings needed for all models.

  Args:
    label_count: How many classes are to be recognized.
    sample_rate: Number of audio samples per second.
    clip_duration_ms: Length of each audio clip to be analyzed.
    window_size_ms: Duration of frequency analysis window.
    window_stride_ms: How far to move in time between frequency windows.
    dct_coefficient_count: Number of frequency bins to use for analysis.

  Returns:
    Dictionary containing common settings.
  """
  desired_samples = int(sample_rate * clip_duration_ms / 1000)
  window_size_samples = int(sample_rate * window_size_ms / 1000)
  window_stride_samples = int(sample_rate * window_stride_ms / 1000)
  length_minus_window = (desired_samples - window_size_samples)
  if length_minus_window < 0:
    spectrogram_length = 0
  else:
    spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
  fingerprint_size = dct_coefficient_count * spectrogram_length
  return {
      'desired_samples': desired_samples,
      'window_size_samples': window_size_samples,
      'window_stride_samples': window_stride_samples,
      'spectrogram_length': spectrogram_length,
      'dct_coefficient_count': dct_coefficient_count,
      'fingerprint_size': fingerprint_size,
      'label_count': label_count,
      'sample_rate': sample_rate,
  }


def create_model(audio_input, model_settings, model_architecture,
                 is_training, runtime_settings=None,
                 norm_binw=False,
                 downsample=1,
                 lock_prefilter=False, add_prefilter_bias=True, use_down_avgfilt=False):
  """Builds a model of the requested architecture compatible with the settings.

  There are many possible ways of deriving predictions from a spectrogram
  input, so this function provides an abstract interface for creating different
  kinds of models in a black-box way. You need to pass in a TensorFlow node as
  the 'fingerprint' input, and this should output a batch of 1D features that
  describe the audio. Typically this will be derived from a spectrogram that's
  been run through an MFCC, but in theory it can be any feature vector of the
  size specified in model_settings['fingerprint_size'].

  The function will build the graph it needs in the current TensorFlow graph,
  and return the tensorflow output that will contain the 'logits' input to the
  softmax prediction process. If training flag is on, it will also return a
  placeholder node that can be used to control the dropout amount.

  See the implementations below for the possible model architectures that can be
  requested.

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    model_architecture: String specifying which kind of model to create.
    is_training: Whether the model is going to be used for training.
    runtime_settings: Dictionary of information about the runtime.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.

  Raises:
    Exception: If the architecture type isn't recognized.
  """
  fingerprint_input = create_spectrogram(audio_input, model_settings, is_training,
                                         downsample=downsample,
                                         add_bias=add_prefilter_bias,
                                         use_avgfilt=use_down_avgfilt)

  if lock_prefilter:
    fingerprint_input = tf.stop_gradient(fingerprint_input)

  #--------------- Prefilter
  # 4 stage + straight
  if model_architecture == 'tinyex3_conv':
    return create_tinyex3_conv_model(fingerprint_input, model_settings,
                                    is_training, filt_k=8)
  elif model_architecture == 'binary3a_conv':
    return create_binary3_conv_model(fingerprint_input, model_settings,
                                    is_training,
                                    no_pool3=True,
                                    normw=norm_binw, stochastic=False)
  elif model_architecture == 'tinyex4_conv':
    return create_tinyex3_conv_model(fingerprint_input, model_settings,
                                    is_training, filt_k=1)
  else:
    raise Exception('model_architecture argument "' + model_architecture +
                    '" not recognized, should be one of "single_fc", "conv",' +
                    ' "low_latency_conv, or "low_latency_svdf"')


def load_variables_from_checkpoint(sess, start_checkpoint):
  """Utility function to centralize checkpoint restoration.

  Args:
    sess: TensorFlow session.
    start_checkpoint: Path to saved checkpoint on disk.
  """
  saver = tf.train.Saver(tf.global_variables())
  saver.restore(sess, start_checkpoint)


def init_variables_from_checkpoint(sess, init_checkpoint, scope=None):
  """Utility function to centralize checkpoint restoration.

  Args:
    sess: TensorFlow session.
    start_checkpoint: Path to saved checkpoint on disk.
  """
  if sess is None:
    assert scope
    tf.train.init_from_checkpoint(init_checkpoint, {scope:scope})
  else:
    if scope:
      global spectrogram_filter
      saver = tf.train.Saver(spectrogram_filter)
      saver.restore(sess, init_checkpoint)
    else:
      saver = tf.train.Saver(tf.global_variables())
      saver.restore(sess, init_checkpoint)


#-----------------------------------------------------
# from r2rt.com
def batch_norm_wrapper(inputs, is_training, scale=True, decay=0.999, epsilon=1e-5):
    dim = inputs.get_shape()
    num_channel = dim[-1]
    scale_v = tf.Variable(tf.ones([num_channel]), name='batchnorm/alpha') if scale else None
    beta = tf.Variable(tf.zeros([num_channel]), name='batchnorm/beta')
    pop_mean = tf.Variable(tf.zeros([num_channel]), trainable=False, name='batchnorm/mean')
    pop_var = tf.Variable(tf.ones([num_channel]), trainable=False, name='batchnorm/var')

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs, list(range(len(dim)-1)), name='moments')
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale_v, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale_v, epsilon)

def _activation_summary(x):
  tensor_name = x.op.name.rsplit('/', 1)[-1]
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


import librosa
import numpy as np

spectrogram_filter = None

def create_spectrogram(audio_input, model_settings, is_training, downsample=1, add_bias=False, use_avgfilt=False):
  sample_size   = model_settings['desired_samples'] // downsample
  filter_width  = model_settings['window_size_samples'] // downsample
  filter_count  = model_settings['dct_coefficient_count']
  filter_stride = model_settings['window_stride_samples'] // downsample
  if downsample == 1:
    #audio_input_3d = tf.reshape(audio_input, [-1, sample_size, 1])
    audio_input_3d = tf.expand_dims(audio_input, -1)
  elif use_avgfilt and downsample == 4:
    sel_filter = tf.convert_to_tensor([[[0.5]], [[0]], [[0.5]], [[0]]], dtype=tf.float32)
    #audio_input_3d = tf.nn.conv1d(audio_input, sel_filter, 4, padding='VALID')
    audio_input_3d = tf.nn.conv1d(tf.expand_dims(audio_input, -1), sel_filter, 4, padding='VALID')
    print(audio_input_3d.get_shape().as_list())
  else:
    print("downsampling to {}".format(downsample))
    #print(audio_input.get_shape().as_list())
    #print(audio_input[:,::downsample].get_shape().as_list())
    audio_input_3d = tf.expand_dims(audio_input[...,::downsample], -1)
    #audio_input_3d = tf.expand_dims(audio_input[...,downsample//2::downsample], -1)    # center pick
  #
  _dct_filters = librosa.filters.dct(filter_count, filter_width)
  print(_dct_filters.shape)
  _dct_filters = np.expand_dims(np.transpose(_dct_filters,(1,0)),1)
  if False:
    dct_filters = tf.convert_to_tensor(_dct_filters, dtype=tf.float32)
    out = tf.nn.conv1d(audio_input_3d, dct_filters, filter_stride, padding='SAME')
    return out
  with tf.variable_scope('freqconv') as scope:
    weights = tf.get_variable('weights', shape=[filter_width, 1, filter_count],
                    initializer=tf.contrib.layers.xavier_initializer())
    #weights = tf.get_variable('weights', initializer=tf.constant_initializer(_dct_filters))
    tf.summary.histogram('weights', weights)
    conv = tf.nn.conv1d(audio_input_3d, weights, filter_stride, padding='VALID')
    global spectrogram_filter
    if False:   # BatchNorm
      norm = batch_norm_wrapper(conv, decay=0.9, epsilon=1e-4,
                                is_training=is_training)
      out = tf.nn.relu(norm, name='relu')
    elif add_bias: # Bias
      bias = tf.get_variable('biases', shape=[filter_count],
                    initializer=tf.constant_initializer(0))
      tf.summary.histogram('biases', bias)
      out = tf.nn.relu(tf.nn.bias_add(conv, bias), name='relu')
      spectrogram_filter = [weights, bias]
    else:
      out = tf.nn.relu(conv, name='relu')
      spectrogram_filter = [weights]
    _activation_summary(out)
    print(weights.get_shape().as_list())
  print(audio_input_3d.get_shape().as_list())
  print(out.get_shape().as_list())
  return out


#-----------------------------------------------------
from binary_ops import binarize, binary_sigmoid, binary_tanh, binary_relu, quatery_relu

def binconv2d(x, W, layer_name, normw=False, depthwise=False, stochastic=False):
  with tf.name_scope(layer_name):
    with tf.name_scope('weights'):
      tf.summary.histogram('histogram', W)
    with tf.name_scope('BinWeights'):
      Wb = binarize(W, normalize=normw, stochastic=stochastic)
      tf.summary.histogram('BinWeights', Wb)
    if depthwise:
      conv_out = tf.nn.depthwise_conv2d(x, Wb, strides=[1, 1, 1, 1], padding='SAME')
    else:
      conv_out = tf.nn.conv2d(x, Wb, strides=[1, 1, 1, 1], padding='SAME')
    tf.summary.histogram('Convout', conv_out)
    output = conv_out   # no bias
  return output

def binfullcon(x, W, layer_name, normw=False, stochastic=False):
  with tf.name_scope(layer_name):
    with tf.name_scope('weights'):
      tf.summary.histogram('histogram', W)
    with tf.name_scope('BinWeights'):
      Wb = binarize(W, normalize=normw, stochastic=stochastic)
      tf.summary.histogram('BinWeights', Wb)
    fc_out = tf.matmul(x, Wb) # no bias
    tf.summary.histogram('Fcout', fc_out)
    #output = fc_out+b
    output = fc_out   # no bias
  return output

#################################################################
# 4 stage + straight
def create_tinyex3_conv_model(fingerprint_input, model_settings,
                              is_training,
                              filt_k=1,
                              depthwise_conv1=False):
  """Builds a convolutional model with low compute requirements.

  (fingerprint_input) -> conv -> pool -> conv -> pool -> conv -> pool -> fc

  """
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  assert(input_frequency_size == 32)
  assert(input_time_size % 32 == 0)
  input_depth = input_time_size // 32
  # HCW  (whole picture in one plane)
  _fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, 32, input_depth, input_frequency_size])
  fingerprint_4d = tf.transpose(_fingerprint_4d, perm=[0, 1, 3, 2])
  print(fingerprint_4d.get_shape().as_list())

  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = tf.get_variable('weights', shape=[3, 3, input_depth, 8*filt_k],
                    initializer=tf.contrib.layers.xavier_initializer())
    conv = tf.nn.conv2d(fingerprint_4d, kernel, [1, 1, 1, 1], padding='SAME')
    norm = batch_norm_wrapper(conv, decay=0.9, epsilon=1e-4,
                              is_training=is_training)
    norm1 = tf.nn.relu(norm, name='relu')
    #_activation_summary(norm1)

  # pool1
  pool1 = tf.nn.max_pool(norm1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = tf.get_variable('weights', shape=[3, 3, 8*filt_k, 8*filt_k],
                    initializer=tf.contrib.layers.xavier_initializer())
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    norm = batch_norm_wrapper(conv, decay=0.9, epsilon=1e-4,
                              is_training=is_training)
    norm2 = tf.nn.relu(norm, name='relu')
    #_activation_summary(norm2)

  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool2')

  # conv3
  with tf.variable_scope('conv3') as scope:
    kernel = tf.get_variable('weights', shape=[3, 3, 8*filt_k, 8*filt_k],
                    initializer=tf.contrib.layers.xavier_initializer())
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    norm = batch_norm_wrapper(conv, decay=0.9, epsilon=1e-4,
                              is_training=is_training)
    norm3 = tf.nn.relu(norm, name='relu')
    #_activation_summary(norm3)

  # pool3
  pool3 = tf.nn.max_pool(norm3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool3')

  # fc4
  if is_training:
    fc_input = tf.nn.dropout(pool3, dropout_prob)
  else:
    fc_input = pool3

  label_count = model_settings['label_count']
  with tf.variable_scope('fc4') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    conv_shape = fc_input.get_shape()
    reshape = tf.reshape(fc_input, [-1, conv_shape[1]*conv_shape[2]*conv_shape[3]])
    dim = reshape.get_shape()[1].value
    weights = tf.get_variable('weights', shape=[dim, label_count],
                    initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable('biases', [label_count], initializer=tf.constant_initializer(0.1))
    fc = tf.matmul(reshape, weights) # no bias
    fc4 = tf.nn.bias_add(fc, biases)
    #_activation_summary(fc4)

  if is_training:
    return fc4, dropout_prob
  else:
    return fc4

def create_binary3_conv_model(fingerprint_input, model_settings,
                              is_training,
                              no_pool3=False,
                              normw=False, stochastic=False):
  """Builds a convolutional model with low compute requirements.

  (fingerprint_input) -> conv -> pool -> conv -> pool -> conv -> conv-> pool -> fc

  """
  print ('normw',normw)
  assert normw == True
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  assert(input_frequency_size == 32)
  assert(input_time_size % 32 == 0)
  input_depth = input_time_size // 32
  # HCW  (whole picture in one plane)
  _fingerprint_4d = tf.reshape(fingerprint_input,
                               [-1, 32, input_depth, input_frequency_size])
  fingerprint_4d = tf.transpose(_fingerprint_4d, perm=[0, 1, 3, 2])
  print(fingerprint_4d.get_shape().as_list())

  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = tf.get_variable('weights', shape=[3, 3, input_depth, 64],
                    initializer=tf.contrib.layers.xavier_initializer())
    conv = binconv2d(fingerprint_4d, kernel, 'conv', normw=normw, stochastic=stochastic)
    norm = batch_norm_wrapper(conv, decay=0.9, epsilon=1e-4,
                              is_training=is_training)
    bin1 = binary_relu(norm)
    _activation_summary(bin1)

  # pool1 (16x16)
  pool1 = tf.nn.max_pool(bin1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = tf.get_variable('weights', shape=[3, 3, 64, 64],
                    initializer=tf.contrib.layers.xavier_initializer())
    conv = binconv2d(pool1, kernel, 'conv', normw=normw, stochastic=stochastic)
    norm = batch_norm_wrapper(conv, decay=0.9, epsilon=1e-4,
                              is_training=is_training)
    bin2 = binary_relu(norm)
    _activation_summary(bin2)

  # pool2 (8x8)
  pool2 = tf.nn.max_pool(bin2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool2')

  # conv3
  with tf.variable_scope('conv3') as scope:
    kernel = tf.get_variable('weights', shape=[3, 3, 64, 64],
                    initializer=tf.contrib.layers.xavier_initializer())
    conv = binconv2d(pool2, kernel, 'conv', normw=normw, stochastic=stochastic)
    norm = batch_norm_wrapper(conv, decay=0.9, epsilon=1e-4,
                              is_training=is_training)
    bin3 = binary_relu(norm)
    _activation_summary(bin3)

  # pool3 (4x4)
  if no_pool3:
    pool3 = bin3
  else:
    pool3 = tf.nn.max_pool(bin3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool3')

  # fc4
  label_count = model_settings['label_count']
  with tf.variable_scope('fc4') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    conv_shape = pool3.get_shape()
    reshape = tf.reshape(pool3, [-1, conv_shape[1]*conv_shape[2]*conv_shape[3]])
    dim = reshape.get_shape()[1].value
    weights = tf.get_variable('weights', shape=[dim, label_count],
                    initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable('biases', [label_count], initializer=tf.constant_initializer(0.1))
    # NOTE: no normalization in last layer
    fc = binfullcon(reshape, weights, 'fc4', normw=False, stochastic=stochastic)
    fc4 = tf.nn.bias_add(fc, biases)
    _activation_summary(fc4)

  if is_training:
    return fc4, dropout_prob
  else:
    return fc4
