import tensorflow as tf
from datasets import flowers
#from datasets import

DATA_DIR='/home/atsg/PycharmProjects/gvh205_py3/tensorflow_slim/datasets/flowers'
slim = tf.contrib.slim

# Selects the 'validation' dataset.
dataset = flowers.get_split('validation', DATA_DIR)

# Creates a TF-Slim DataProvider which reads the dataset in the background
# during both training and testing.
provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
[image, label] = provider.get(['image', 'label'])
kk=1