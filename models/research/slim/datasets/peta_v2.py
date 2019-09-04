from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

slim = tf.contrib.slim

_FILE_PATTERN = '%s-*.tfrecord'
_SPLITS_TO_SIZES = {'train': 29649, 'test': 1376}
_NUM_CLASSES = 31

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A list of labels',
    'adj': 'A list of one, one per each label',
}

_LABELS_FILENAME = '/home/atsg/PycharmProjects/gvh205_py3/tensorflow_slim/pytorch/data/PETA_tfrecord/label_map.txt'

def read_label_file(label_file_path):
    with tf.gfile.Open(label_file_path, 'rb') as f:
        lines = f.read().decode()
    lines = lines.split('\n')
    lines = filter(None, lines)

    labels_to_class_names = {}
    for line in lines:
        parts = line.split(' ')
        labels_to_class_names[int(parts[0])] = parts[1]
    return labels_to_class_names

def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    if split_name not in _SPLITS_TO_SIZES:
        raise ValueError('split name %s was not recognized.' % split_name)

    if not file_pattern:
        file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader

    # Features in HICO TFRecords
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/class/label': tf.FixedLenFeature([], tf.string),
        #'image/class/adj': tf.VarLenFeature(dtype=tf.string),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
        #'adj': slim.tfexample_decoder.Tensor('image/class/adj'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = read_label_file(_LABELS_FILENAME)

    return slim.dataset.Dataset(
            data_sources=file_pattern,
            reader=reader,
            decoder=decoder,
            num_samples=_SPLITS_TO_SIZES[split_name],
            items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
            num_classes=_NUM_CLASSES,
            labels_to_names=labels_to_names)
