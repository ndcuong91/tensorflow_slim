from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
from datetime import datetime
import math
from utils_image_classifier import create_dir
from datasets import dataset_factory
from preprocessing import preprocessing_factory
from nets import nets_factory

import config_image_classifier as config

slim = tf.contrib.slim

image_size = config.input_size
dataset_name=config.dataset_name
dataset_dir=config.dataset_dir
dataset_split='train'
checkpoint_path=config.checkpoint_train_path
model_name=config.model_name
quant_delay=config.quant_delay
output_dir=config.output_dir
log_every_n_steps=config.log_every_n_steps
trainable_scopes=config.trainable_scopes
num_thread=config.num_thread
batch_size=config.batch_size
num_epoch=config.num_epoch
lr=config.lr


tf.app.flags.DEFINE_string(
    'model_name', model_name, 'The name of the architecture to train.')
tf.flags.DEFINE_integer('batch_size', batch_size, 'Batch size')
tf.flags.DEFINE_integer('epochs', num_epoch, 'Number of training epochs')
tf.flags.DEFINE_float('learning_rate', lr, 'Initial learning rate')
tf.flags.DEFINE_string('dataset_dir', dataset_dir,
                       'The directory where the dataset files are stored')
tf.flags.DEFINE_string('checkpoint_path', checkpoint_path,
                       'The directory where the pretrained model is stored')
tf.app.flags.DEFINE_string(
    'dataset_name', dataset_name, 'The name of the dataset to load.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', dataset_split, 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'output_dir', output_dir,
    'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_integer(
    'quantize_delay', quant_delay,
    'Number of steps to start quantized training. Set to -1 would disable '
    'quantized training.')
tf.app.flags.DEFINE_integer(
    'log_every_n_steps', log_every_n_steps,
    'The frequency with which logs are print.')
tf.app.flags.DEFINE_string(
    'trainable_scopes', trainable_scopes,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')
tf.app.flags.DEFINE_integer(
    'num_readers', num_thread,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', num_thread,
    'The number of threads used to create the batches.')

FLAGS = tf.app.flags.FLAGS


def get_init_fn():
    checkpoint_exclude_scopes = ['resnet_v1_50/logits','resnet_v1_50/AuxLogits']
    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    return slim.assign_from_checkpoint_fn(FLAGS.checkpoint_path,variables_to_restore, ignore_missing_vars=True)


def _get_variables_to_train():
  """Returns a list of variables to train.

  Returns:
    A list of variables to train by the optimizer.
  """
  if FLAGS.trainable_scopes =='':
    print('Train all variables!')
    return tf.trainable_variables()
  else:
    print('Train some variables!')
    scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

  variables_to_train = []
  for scope in scopes:
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    variables_to_train.extend(variables)
  return variables_to_train

def logging(dir):
    import logging

    # get TF logger
    log = logging.getLogger('tensorflow')
    log.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(dir,'training_tf.log'))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    log.addHandler(fh)

def main(_):

    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.ERROR)

        # Select the dataset
        dataset = dataset_factory.get_dataset(FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)
        data_provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=FLAGS.num_readers,
            common_queue_capacity=20 * FLAGS.batch_size,
            common_queue_min=10 * FLAGS.batch_size)

        image, label = data_provider.get(['image', 'label'])

        label = tf.decode_raw(label, tf.float32)

        label = tf.reshape(label, [dataset.num_classes])

        # Preprocess images
        preprocessing_name = 'deepmar'
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(preprocessing_name,is_training=True)
        image = image_preprocessing_fn(image, image_size, image_size)

        # Training bathes and queue
        images, labels = tf.train.batch(
            [image, label],
            batch_size=FLAGS.batch_size,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity=5 * FLAGS.batch_size)

        # Create the model
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=dataset.num_classes,
            weight_decay=FLAGS.weight_decay,
            is_training=True)

        logits, _ = network_fn(images)

        #predictions = tf.nn.sigmoid(logits, name='prediction')

        if FLAGS.quantize_delay >= 0:
            print('Train with quantization-aware training')
            tf.contrib.quantize.create_training_graph(
                quant_delay=0)

        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        for loss in tf.get_collection(tf.GraphKeys.LOSSES):
            summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(cross_entropy)

        # Add summaries
        tf.summary.scalar('loss', loss)

        # Variables to train.
        variables_to_train = _get_variables_to_train()

        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        #optimizer = tf.train.cosine_decay(learning_rate=FLAGS.learning_rate, decay_steps = 1000)

        train_op = slim.learning.create_train_op(loss, optimizer, variables_to_train=variables_to_train)

        num_batches = math.ceil(data_provider.num_samples() / float(FLAGS.batch_size))
        num_steps = FLAGS.epochs * int(num_batches)

        # Kicks off the training. #
        folder = FLAGS.model_name + '_' + str(image_size) + '_' + FLAGS.dataset_name
        date_time = datetime.now().strftime('%Y-%m-%d_%H.%M')
        log_dir = os.path.join(FLAGS.output_dir, folder, date_time)

        if (FLAGS.checkpoint_path == ''):
            create_dir(log_dir)
        else:
            log_dir = os.path.dirname(FLAGS.checkpoint_path)

        logging(log_dir)
        tf.logging.set_verbosity(tf.logging.INFO)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}
        tf.logging.info(FLAGS.flag_values_dict())
        saver = tf.train.Saver(max_to_keep=100)

        print('Begin training resnet_v1_50 model')
        slim.learning.train(
            train_op,
            logdir=log_dir,
            init_fn=get_init_fn(),
            number_of_steps=num_steps,
            save_summaries_secs=300,
            save_interval_secs=300,
            log_every_n_steps=FLAGS.log_every_n_steps,
            saver=saver
        )


if __name__ == '__main__':
    tf.app.run()