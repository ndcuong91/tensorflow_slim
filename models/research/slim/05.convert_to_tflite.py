import tensorflow.contrib.lite as tflite
import tensorflow as tf

import tensorflow as tf
gf = tf.GraphDef()

import config_image_classifier as config

input_file=config.output_file
input_tensor='input'
output_tensor='MobilenetV1/Predictions/Softmax'
DATASET_NAME=config.DATASET_NAME
MODEL_NAME=config.MODEL_NAME


m_file = open(input_file,'rb')
gf.ParseFromString(m_file.read())
for n in gf.node:
    print( n.name )

tensor = n.op

#
# converter = tflite.TFLiteConverter.from_frozen_graph(input_file,
#                                                       [input_tensor],
#                                                       [output_tensor])
#
# #converter.output_file = "./tflite_model.tflite"
# converter.inference_type = tflite.constants.QUANTIZED_UINT8
# converter.inference_input_type = tflite.constants.QUANTIZED_UINT8
# converter.quantized_input_stats = {input_tensor: (0.0, 255.0)}
# # converter.default_ranges_stats = (0, 6)
# # converter.inference_output_type = tf.float32
#
# converter.dump_graphviz_dir = './'
#
# converter.dump_graphviz_video = False
#
# flatbuffer = converter.convert()
#
# with open(MODEL_NAME+'_'+DATASET_NAME+'.tflite', 'wb') as outfile:
#     outfile.write(flatbuffer)
