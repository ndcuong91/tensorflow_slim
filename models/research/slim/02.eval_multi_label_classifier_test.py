from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import math
from datasets import dataset_factory
from preprocessing import preprocessing_factory
from nets import nets_factory

import config_image_classifier as config


slim = tf.contrib.slim

image_size = config.input_size
dataset_name=config.dataset_name
dataset_dir=config.dataset_dir
dataset_split='test'
checkpoint_path=config.checkpoint_eval_path
model_name=config.model_name
quant_delay=config.quant_delay
output_dir=config.output_dir

tf.flags.DEFINE_integer('batch_size', 32, 'Batch size')

tf.app.flags.DEFINE_string(
    'model_name', model_name, 'The name of the architecture to test.')
tf.flags.DEFINE_string('dataset_dir', dataset_dir,
                       'The directory where the dataset files are stored')
tf.flags.DEFINE_string('checkpoint_path', checkpoint_path,
                       'The directory where the pretrained model is stored')
tf.flags.DEFINE_integer('num_classes', 31,
                        'Number of classes')
tf.app.flags.DEFINE_string(
    'dataset_name', dataset_name, 'The name of the dataset to load.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', dataset_split, 'The name of the train/test split.')

FLAGS = tf.app.flags.FLAGS

num_class=31
total_samples=1376

# reference: https://github.com/broadinstitute/keras-rcnn/issues/6
def cal_mAP(y_pred, y_true):
    true_preds = []
    for i in range(num_class):
        true_preds.append(0)

    for i in range(total_samples):
        for j in range(31):
            true_val=y_true[i][j]
            pred_val=y_pred[i][j]

            if(pred_val<0.5):
                pred_val=0
            elif(pred_val>0.5):
                pred_val=1
            else:
                true_preds[j]+=1
                continue
            if(pred_val==int(true_val)):
                true_preds[j]+=1
    accum_acc = 0
    for i in range(num_class):
        acc = float(true_preds[i]) / float(total_samples)
        accum_acc += acc
        print('Class ', i, ':', acc)
    print('Final acc:', accum_acc / float(num_class))
    return 0



def calculate_mAP(y_pred, y_true):
    num_classes = y_true.shape[1]
    average_precisions = []

    for index in range(FLAGS.num_classes):
        pred = y_pred[:, index]
        label = y_true[:, index]

        sorted_indices = np.argsort(-pred)
        sorted_pred = pred[sorted_indices]
        sorted_label = label[sorted_indices]

        tp = (sorted_label == 1)
        fp = (sorted_label == 0)

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)

        npos = np.sum(sorted_label)

        recall = tp * 1.0 / npos

        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        precision = tp * 1.0 / np.maximum((tp + fp), np.finfo(np.float64).eps)

        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        average_precisions.append(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))

    print(average_precisions)
    mAP = np.mean(average_precisions)

    return mAP


def main(_):
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)

        # Select the dataset
        dataset = dataset_factory.get_dataset(FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)
        data_provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=1,
            common_queue_capacity=20 * FLAGS.batch_size,
            common_queue_min=10 * FLAGS.batch_size,
            shuffle=False)

        image, label = data_provider.get(['image', 'label'])

        label = tf.decode_raw(label, tf.float32)

        label = tf.reshape(label, [FLAGS.num_classes])

        # Preprocess images
        preprocessing_name = FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(preprocessing_name, is_training=False)
        image = image_preprocessing_fn(image, image_size, image_size)

        # Training bathes and queue
        images, labels = tf.train.batch(
            [image, label],
            batch_size=FLAGS.batch_size,
            num_threads=1,
            capacity=5 * FLAGS.batch_size,
            allow_smaller_final_batch=True)

        # Create the model
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=dataset.num_classes,
            is_training=False)

        logits, _ = network_fn(images)

        predictions = tf.nn.sigmoid(logits, name='prediction')

        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(cross_entropy)

        correct_prediction = tf.equal(tf.round(predictions), labels)

        # Mean accuracy over all labels:
        # http://stackoverflow.com/questions/37746670/tensorflow-multi-label-accuracy-calculation
        accuracy = tf.cast(correct_prediction, tf.float32)
        mean_accuracy = tf.reduce_mean(accuracy)

        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        init_fn = slim.assign_from_checkpoint_fn(
            checkpoint_path,
            slim.get_variables_to_restore())

        num_samples=data_provider.num_samples()
        num_batches = math.ceil(data_provider.num_samples() / float(FLAGS.batch_size))

        prediction_list = []
        label_list = []
        count = 0

        conf = tf.ConfigProto(device_count={'GPU': 0})  #
        with tf.Session(config=conf) as sess:
            with slim.queues.QueueRunners(sess):
                sess.run(tf.local_variables_initializer())
                init_fn(sess)

                for step in range(int(num_batches)):
                    np_loss, np_accuracy, np_logits, np_prediction, np_labels = sess.run(
                        [loss, mean_accuracy, logits, predictions, labels])

                    prediction_list.append(np_prediction)
                    label_list.append(np_labels)

                    count += np_labels.shape[0]

                    print('Step {}, count {}, loss: {}'.format(step, count, np_loss))

        prediction_arr = np.concatenate(prediction_list, axis=0)
        label_arr = np.concatenate(label_list, axis=0)

        mAP = cal_mAP(prediction_arr, label_arr)
        #print('mAP score: {}'.format(mAP))


if __name__ == '__main__':
    tf.app.run()