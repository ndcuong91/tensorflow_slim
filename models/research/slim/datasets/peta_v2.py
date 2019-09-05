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


def read_label_file():
    labels_to_class_names = {}
    labels_to_class_names[0]='personalLess30'
    labels_to_class_names[1] ='personalLess45'
    labels_to_class_names[2] ='personalLess60'
    labels_to_class_names[3] ='personalLarger60'
    labels_to_class_names[4] ='personalFemale'
    labels_to_class_names[5] ='personalMale'
    labels_to_class_names[6] ='carryingNothing'
    labels_to_class_names[7] ='carryingMessengerBag'
    labels_to_class_names[8] ='carryingBackpack'
    labels_to_class_names[9] ='hairLong'
    labels_to_class_names[10] ='hairShort'
    labels_to_class_names[11] ='lowerBodyCasual'
    labels_to_class_names[12] ='lowerBodyFormal'
    labels_to_class_names[13] ='lowerBodyJeans'
    labels_to_class_names[14] ='lowerBodyTrousers'
    labels_to_class_names[15] ='upperBodyTshirt'
    labels_to_class_names[16] ='upperBodyOther'
    labels_to_class_names[17] ='upperBodyCasual'
    labels_to_class_names[18] ='upperBodyFormal'
    labels_to_class_names[19] ='upperBodyLongSleeve'
    labels_to_class_names[20] ='upperBodyShortSleeve'
    labels_to_class_names[21] ='accessoryHat'
    labels_to_class_names[22] ='accessoryMuffler'
    labels_to_class_names[23] ='accessoryNothing'
    labels_to_class_names[24] ='lowerBodyBlack'
    labels_to_class_names[25] ='lowerBodyBlue'
    labels_to_class_names[26] ='lowerBodyGrey'
    labels_to_class_names[27] ='upperBodyBlack'
    labels_to_class_names[28] ='upperBodyBlue'
    labels_to_class_names[29] ='upperBodyGrey'
    labels_to_class_names[30] ='upperBodyWhite'

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
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = read_label_file()

    return slim.dataset.Dataset(
            data_sources=file_pattern,
            reader=reader,
            decoder=decoder,
            num_samples=_SPLITS_TO_SIZES[split_name],
            items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
            num_classes=_NUM_CLASSES,
            labels_to_names=labels_to_names)
