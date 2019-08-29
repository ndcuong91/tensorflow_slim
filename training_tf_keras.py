import tensorflow as tf
tf.enable_eager_execution()
from tensorflow import keras
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
import numpy as np

print(tf.__version__)

checkpoint_path='outputs/test/res50_qt.ckpt'
output_file='outputs/test/res50_qt_frozen.pb'
output_tffile='outputs/test/res50_qt_frozen.tflite'
input_node='resnet50_input'
output_node='dense/Softmax'
fine_tuning=True
size=112
input_shape=(size,size,3)

def build_keras_model():
    if(fine_tuning):
        base_model = keras.applications.resnet50.ResNet50(input_shape=input_shape, weights=None, include_top=False)
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        prediction_layer = Dense(10, activation='softmax')
        model = keras.Sequential([
            base_model,
            global_average_layer,
            prediction_layer
        ])
    else:
        model= keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(10, activation='softmax')
        ])

    return model

def train1():
    print('Train----------------------------------')
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    if(fine_tuning):
        train_images=np.zeros(shape=(1000,size,size,3))
        train_labels=np.zeros(shape=(1000))

        test_images=np.zeros(shape=(600,size,size,3))
        test_labels=np.zeros(shape=(600))
    # train
    train_graph = tf.Graph()
    train_sess = tf.Session(graph=train_graph)

    keras.backend.set_session(train_sess)
    with train_graph.as_default():
        train_model = build_keras_model()
        tf.contrib.quantize.create_training_graph(input_graph=train_graph, quant_delay=0)
        #writer = tf.summary.FileWriter("outputs", train_sess.graph)
        train_sess.run(tf.global_variables_initializer())

        train_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        train_model.fit(train_images, train_labels, epochs=1)
        #writer.close()

        # save graph and checkpoints
        saver = tf.train.Saver()
        saver.save(train_sess, checkpoint_path)
    print ('Finish save checkpoint file', checkpoint_path)

def eval1():
    print('Eval----------------------------------')
    eval_graph = tf.Graph()
    eval_sess = tf.Session(graph=eval_graph)
    keras.backend.set_session(eval_sess)

    with eval_graph.as_default():
        keras.backend.set_learning_phase(0)
        eval_model = build_keras_model()
        tf.contrib.quantize.create_eval_graph(input_graph=eval_graph)
        eval_graph_def = eval_graph.as_graph_def()
        saver = tf.train.Saver()
        saver.restore(eval_sess, checkpoint_path)

        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            eval_sess,
            eval_graph_def,
            [eval_model.output.op.name]
        )
        with open(output_file, 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())
    print ('Finish create frozen file', output_file)

import tensorflow.contrib.lite as tflite
def convert():
    converter = tflite.TFLiteConverter.from_frozen_graph(output_file, [input_node], [output_node])

    # converter.output_file = "./tflite_model.tflite"
    converter.inference_type = tflite.constants.QUANTIZED_UINT8
    converter.inference_input_type = tflite.constants.QUANTIZED_UINT8
    converter.quantized_input_stats = {input_node: (0.0, 255.0)}
    # converter.default_ranges_stats = (0, 6)
    # converter.inference_output_type = tf.float32

    converter.dump_graphviz_dir = './'

    converter.dump_graphviz_video = False

    flatbuffer = converter.convert()

    with open(output_tffile, 'wb') as outfile:
        outfile.write(flatbuffer)
    print ('Finish convert frozen file',output_file)
    print ('to tflite file',output_tffile)

def train_cat_and_dog():
    import tensorflow_datasets as tfds
    #tfds.disable_progress_bar()
    SPLIT_WEIGHTS = (8, 1, 1)
    splits = tfds.Split.TRAIN.subsplit(weighted=SPLIT_WEIGHTS)

    (raw_train, raw_validation, raw_test), metadata = tfds.load(
        'cats_vs_dogs', split=list(splits),
        with_info=True, as_supervised=True)
    IMG_SIZE = 160  # All images will be resized to 160x160

    def format_example(image, label):
        image = tf.cast(image, tf.float32)
        image = (image / 127.5) - 1
        image = tf.image.resize_images(image, (IMG_SIZE, IMG_SIZE))
        return image, label

    train = raw_train.map(format_example)
    validation = raw_validation.map(format_example)
    test = raw_test.map(format_example)

    BATCH_SIZE = 32
    SHUFFLE_BUFFER_SIZE = 1000

    train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    validation_batches = validation.batch(BATCH_SIZE)
    test_batches = test.batch(BATCH_SIZE)

    for image_batch, label_batch in train_batches.take(1):
        pass
    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

    # Create the base model from the pre-trained model MobileNet V2
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')

    feature_batch = base_model(image_batch)
    print(feature_batch.shape)

    base_model.trainable = False

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    prediction_layer = keras.layers.Dense(1)
    prediction_batch = prediction_layer(feature_batch_average)
    print(prediction_batch.shape)


    base_learning_rate = 0.0001


    # train
    train_graph = tf.Graph()
    train_sess = tf.Session(graph=train_graph)

    keras.backend.set_session(train_sess)

    with train_graph.as_default():
        model = tf.keras.Sequential([
            base_model,
            global_average_layer,
            prediction_layer
        ])


        tf.contrib.quantize.create_training_graph(input_graph=train_graph, quant_delay=100)
        train_sess.run(tf.global_variables_initializer())

        model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
        initial_epochs = 10
        model.fit(train_batches,
                  epochs=initial_epochs,
                  validation_data=validation_batches)

        saver = tf.train.Saver()
        saver.save(train_sess, '/home/duycuong/PycharmProjects/research_py3/tensorflow_slim/outputs/test.ckpt')

if __name__ == '__main__':
    #train_cat_and_dog()
    train1()
    eval1()
    convert()