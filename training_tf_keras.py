import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.__version__)
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

def build_keras_model():
    return keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(10, activation='softmax')
    ])


# train
train_graph = tf.Graph()
train_sess = tf.Session(graph=train_graph)

keras.backend.set_session(train_sess)
with train_graph.as_default():
    train_model = build_keras_model()

    tf.contrib.quantize.create_training_graph(input_graph=train_graph, quant_delay=100)
    train_sess.run(tf.global_variables_initializer())

    train_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    train_model.fit(train_images, train_labels, epochs=5)

    # save graph and checkpoints
    saver = tf.train.Saver()
    saver.save(train_sess, 'outputs')