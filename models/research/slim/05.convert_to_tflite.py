import tensorflow.contrib.lite as tflite
import tensorflow as tf

import config_image_classifier as config

input_file=config.output_file
input_node=config.input_node
output_node=config.output_node
output_tffile=config.output_tflite

def print_all_nodes():
    gf = tf.GraphDef()
    m_file = open(input_file, 'rb')
    gf.ParseFromString(m_file.read())
    for n in gf.node:
        print(n.name)

def convert():
    converter = tflite.TFLiteConverter.from_frozen_graph(input_file, [input_node], [output_node])

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
    print ('Finish convert frozen file',input_file)
    print ('to tflite file',output_tffile)

if __name__ == '__main__':
    print('Program begin')
    #print_all_nodes()
    convert()