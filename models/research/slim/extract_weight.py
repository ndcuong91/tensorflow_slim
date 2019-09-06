import tensorflow as tf
from tensorflow.python.platform import gfile
import sys, os
from google.protobuf import text_format
import config_image_classifier as config
from tensorflow.python.framework import tensor_util
from tensorflow.contrib import graph_editor as ge


## In tensorflow the weights are also stored in constants ops
## So to get the values of the weights, you need to run the constant ops
## It's a little bit anti-intution, but that's the way they do it

#construct a GraphDef
pbFile = '../../../outputs/tflite/AnG_model_frozen.pb'
pbtxtFile = '../../../outputs/tflite/AnG_model_graph.pbtxt'

def pbtxt_to_graphdef(filename):
    dir_name=os.path.dirname(filename)
    base_name=os.path.basename(filename)
    with open(filename, 'r') as f:
        graph_def = tf.GraphDef()
        file_content = f.read()
        text_format.Merge(file_content, graph_def)
        tf.import_graph_def(graph_def, name='')
        tf.train.write_graph(graph_def, dir_name, base_name.replace('.pbtxt','.pb'), as_text=False)

def graphdef_to_pbtxt(filename):
    dir_name=os.path.dirname(filename)
    base_name=os.path.basename(filename)
    with gfile.FastGFile(filename,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        tf.train.write_graph(graph_def, dir_name, base_name.replace('.pb','.pbtxt'), as_text=True)
        print ('Finish convert file',filename, 'to .pbtxt file!')
    return

def extract_weight():
    graph_def = tf.GraphDef()
    with open(pbFile, 'rb') as f:
        graph_def.ParseFromString(f.read())

    # import the GraphDef to the global default Graph
    tf.import_graph_def(graph_def, name='')

    # extract all the constant ops from the Graph
    # and run all the constant ops to get the values (weights) of the constant ops
    constant_values = {}
    with tf.Session() as sess:
        constant_ops = [op for op in sess.graph.get_operations() if op.type == "Const"]
        for constant_op in constant_ops:
            value = sess.run(constant_op.outputs[0])
            constant_values[constant_op.name] = value

            # In most cases, the type of the value is a numpy.ndarray.
            # So, if you just print it, sometimes many of the values of the array will
            # be replaced by ...
            # But at least you get an array to python object,
            # you can do what other you want to save it to the format you want
            if (constant_op.name == 'resnet_v1_50/logits/biases'):
                print(constant_op.name, value)

def modify_graph_backup(output_file='../../../outputs/tflite/test.pb'):
    #init old graph and parse from saved frozen model
    old_graph_def = tf.GraphDef()
    with open(pbFile, 'rb') as f:
        old_graph_def.ParseFromString(f.read())

    # import the GraphDef to the global default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(old_graph_def, name='')

    #init new graph
    new_graph_def = tf.GraphDef()

    with tf.Session(graph=graph) as sess:
        for old_node in sess.graph_def.node:
            #print('node type:', old_node.op)
            #print('node name:', old_node.name)
            # add new node for new graph
            new_node = new_graph_def.node.add()

            if old_node.op == 'Sigmoid':
                new_node.op = 'Tanh'
                new_node.name = old_node.name
                for i in old_node.input:
                    new_node.input.extend([i])
                print ('Finish convert sigmoid to tanh')
            else:
                #copy another node
                new_node.CopyFrom(old_node)
                print ('Copy node type:', new_node.op)
                print ('Copy node name:', new_node.name)
    # Write GraphDef to file if output path has been given.
    if(output_file!=''):
        with gfile.GFile(output_file, "wb") as f:
            f.write(new_graph_def.SerializeToString())
            print ('Save new graph to file:', output_file)

def modify_graph(output_file='../../../outputs/tflite/test.pb'):
    #init old graph and parse from saved frozen model
    old_graph_def = tf.GraphDef()
    with open(pbFile, 'rb') as f:
        old_graph_def.ParseFromString(f.read())

    # import the GraphDef to the global default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(old_graph_def, name='')

    #init new graph
    new_graph_def = tf.GraphDef()

    with tf.Session(graph=graph) as sess:
        dense_weight=None
        dense_1_weight=None
        for old_node in sess.graph_def.node:
            # add new node for new graph
            if '/dense/' in old_node.name or '/Softmax' in old_node.name:
                if old_node.name=='model/classifier_block/dense/kernel':
                    value = old_node.attr['value'].tensor
                    dense_weight = tensor_util.MakeNdarray(value)
                    print('Finish copy weight from dense to ndarray!')
                print('ignore node name:', old_node.name, 'type', old_node.op)
            else:
                if old_node.name=='model/classifier_block/dense_1/kernel':
                    value = old_node.attr['value'].tensor
                    dense_1_weight=tensor_util.MakeNdarray(value)
                    print('Finish copy weight from dense_1 to ndarray!')
                    modify_node=create_new_node(dense_1_weight,dense_weight,old_node)
                    new_node = new_graph_def.node.add()
                    new_node.CopyFrom(modify_node)
                    print('Finish modify node',old_node.name)
                else:
                    new_node = new_graph_def.node.add()
                    #copy another node
                    new_node.CopyFrom(old_node)
                    print ('Copy node type:', new_node.op)
                    print ('Copy node name:', new_node.name)

    # Write GraphDef to file if output path has been given.
    if(output_file!=''):
        with gfile.GFile(output_file, "wb") as f:
            f.write(new_graph_def.SerializeToString())
            print ('Save new graph to file:', output_file)


def create_new_node(input_weight_a, intput_weight_b, old_node):
    import numpy as np
    #w_init = np.random.randn(131072, 103).astype(np.float32)

    merge_weight=np.concatenate((input_weight_a,intput_weight_b), axis=1)
    #w = tf.Variable(tf.convert_to_tensor(w_init))
    tensor_proto=tf.make_tensor_proto(merge_weight)

    new_node = tf.NodeDef(name=old_node.name, op='Const',
                      attr={'value': tf.AttrValue(tensor=tensor_proto),
                            'dtype': tf.AttrValue(type='DT_FLOAT')})
    #new_node.input.extend([' model/classifier_block/flatten', 'model/classifier_block/dense_1/kernel/read'])
    return new_node

def useful_function():
    kk=1
    # node = tf.NodeDef()
    # node.name = 'MySub'
    # node.op = 'Sub'
    # node.input.extend(['MyConstTensor', 'conv2'])
    # node.attr["key"].s = 'T'
    # for k in old_node.attr.keys():
    #     attr_val = old_node.attr[k].tensor


if __name__=='__main__':
    #extract_weight()
    #modify_graph_backup(output_file='')
    #graphdef_to_pbtxt(pbFile)
    #create_new_node()
    modify_graph()
