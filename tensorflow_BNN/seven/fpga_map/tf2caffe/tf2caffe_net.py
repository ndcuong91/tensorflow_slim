#!/bin/env python
"""parse tensorflow checkpoint"""
"""
  Inputs:
    - tensorflow checkpoint
    - caffe model file
  Outputs:
    - caffe weights file

  match tensor name in TF and layer name in Caffe
"""

import tensorflow as tf
from google.protobuf import text_format

import sys
import os
#sys.path.insert(0, os.path.join(os.environ["CAFFE_ROOT"], "python"))
sys.path.append(os.path.join("/home/labuser/ML/tools/caffe1/", "python"))
from caffe import layers as cl
from caffe import params as cp
import caffe
import argparse
import logging

# TF API         node.op
# nn.conv2d   -> Conv2D
# nn.bias_add -> BiasAdd
# nn.relu     -> Relu
# nn.max_pool -> MaxPool
# nn.avg_pool -> 
# nn.lrn      -> LRN
# nn.reshape  -> Reshape
# nn.matmul   -> MatMul
# nn.add      -> Add
# +           -> Add

def convert(pb_file, caffe_model):
    gdef = tf.GraphDef()
    if pb_file.endswith('pbtxt'):
        with tf.gfile.FastGFile(pb_file, 'r') as f:
            text_format.Merge(f.read(), gdef)
    else:
        with tf.gfile.FastGFile(pb_file, 'rb') as f:
            gdef.ParseFromString(f.read())
    prev_node = None
    for node in gdef.node:
        if '/gradients/' in node.name:
            continue
        if node.op == 'Conv2D':
            assert(prev_node)
            node_name = node.name.rsplit('/',1)[0]     # ex) conv1/Conv2D -> conv1
            fields = dict()
            w_node = find_variable_node(gdef, node.input[1])
            ddim = [int(a.size) for a in w_node.attr['shape'].shape.dim]
            fields['num_output'] = ddim[-1]
            fields['kernel_size'] = ddim[0]
            if 'strides' in node.attr:
                fields['stride'] = node.attr['strides'].list.i[1]
            if 'SAME' in str(node.attr['padding'].s):
                fields['pad'] = ddim[0] // 2
            elif 'VALID' in str(node.attr['padding'].s):
                fields['pad'] = 0
            else:
                assert False, str(node.attr['padding'].s)
            next_node = find_next_node_by_name(gdef, node.name)
            if next_node.op != 'BiasAdd':
                fields['bias_term'] = False
            logging.info("------ {}(Convolution)".format(node_name))
            prev_node = cl.Convolution(prev_node, name=node_name, **fields)
        elif node.op == 'BiasAdd':
            #prev_node = find_node_by_name(gdef, node.input[0])
            #assert(prev_node.op == 'Conv2D')
            pass
        elif node.op == 'MatMul':
            assert(prev_node)
            node_name = node.name.rsplit('/',1)[0]     # local3/MatMul -> local3
            fields = dict()
            w_node = find_variable_node(gdef, node.input[1])
            ddim = [int(a.size) for a in w_node.attr['shape'].shape.dim]
            fields['num_output'] = ddim[-1]
            next_node = find_next_node_by_name(gdef, node.name)
            if next_node.op != 'BiasAdd':
                fields['bias_term'] = False
            logging.info("------ {}(FullConnect)".format(node_name))
            prev_node = cl.InnerProduct(prev_node, name=node_name, **fields)
        elif node.op == 'Add':
            if '/Initializer/' in node.name:
                pass
            elif '/batchnorm/' in node.name:
                pass
            else:
                #cur_node = find_node_by_name(gdef, node.input[0])
                #assert(cur_node.op == 'MatMul')
                pass
        elif node.op == 'Relu':
            fields = {'in_place':True}
            logging.info("----- {}(ReLU)".format(node.name))
            prev_node = cl.ReLU(prev_node, name=node.name, **fields)
        elif node.op == 'Reshape':
            pass
        elif node.op == 'MaxPool':
            fields = {'pool':cp.Pooling.MAX}
            if 'ksize' in node.attr:
                fields['kernel_size'] = node.attr['ksize'].list.i[1]
            if 'strides' in node.attr:
                fields['stride'] = node.attr['strides'].list.i[1]
            logging.info("----- {}(MaxPooling)".format(node.name))
            prev_node = cl.Pooling(prev_node, name=node.name, **fields)
        elif node.op == 'LRN':
            fields = {
                'local_size': node.attr['depth_radius'].i+1,
                'alpha':  node.attr['alpha'].f,
                'beta': node.attr['beta'].f,
            }
            prev_node = cl.LRN(prev_node, name=node.name, **fields)
        elif node.op == 'Placeholder':
            if node.name == 'image_input':
                logging.info("----- {}(Input)".format(node.name))
                fields = dict(shape={"dim": [1, 3, 224, 224]})
                prev_node = cl.Input(name=node.name, **fields)
            elif node.name == 'audio_input':
                logging.info("----- {}(Input)".format(node.name))
                fields = dict(shape={"dim": [1, 1, 4160, 1]})
                prev_node = cl.Input(name=node.name, **fields)
            else:
                logging.info("skip placeholder, {}".format(node.name))
        elif node.op == 'Rsqrt':
            if '/batchnorm/' in node.name:
                node_name = node.name.rsplit('/',1)[0]     # conv1/batchnorm/Rsqrt -> conv1/batchnorm
                fields = dict()
                prev_node = cl.BatchNorm(prev_node, name=node_name, **fields)
                fields = {'bias_term':True}
                prev_node = cl.Scale(prev_node, name=node_name+'/scale', **fields)
        elif node.op == 'DepthwiseConv2dNative':
            assert False, "{} is not supported".format(node.op)
        elif node.op == 'ExpandDims':
            if prev_node is None:
                logging.info("----- {}(Input)".format(node.name))
                fields = dict(shape={"dim": [1, 3, 24, 24]})
                prev_node = cl.Input(name=node.name, **fields)
        elif node.op in ['InTopK']:
            pass
        elif node.op in ['Assign', 'VariableV2', 'Const', 'Cast']:
            pass
        elif node.op in ['Mul', 'Sub', 'Identity', 'Mean', 'Equal', 'NoOp', 'L2Loss', 'TruncatedNormal']:
            pass
        elif node.op in ['ReadFile', 'DecodePng']:
            pass
        elif node.op in ['SaveV2', 'RestoreV2']:
            pass
        elif node.op in ['MergeSummary', 'ScalarSummary', 'HistogramSummary']:
            pass
        else:
            logging.debug('    '+node.op)
            logging.debug('    '+node.name)    # output
            logging.debug('    '+', '.join(node.input))
            logging.debug('    '+str(list(node.attr.keys())))
            pass
    with open(caffe_model, 'w') as f:
        f.write(str(prev_node.to_proto()))

def find_node_by_name(gdef, name):
    for node in gdef.node:
        if name == node.name:
            return node
    return None

def find_next_node_by_name(gdef, name):
    for node in gdef.node:
        if name in node.input:
            return node
    return None

def find_variable_node(gdef, name):
    node = find_node_by_name(gdef, name)
    if node.op in ['VariableV2', 'Placeholder']:
        return node
    for top_name in node.input:
        w_node = find_variable_node(gdef, top_name)
        if w_node: return w_node
    return None

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Tensorflow to Caffe Network')
  parser.add_argument('--input', type=str, default="test.pbtxt", help='pb/pbtxt generated from TF')
  parser.add_argument('--output', type=str, default="test.proto", help='prototxt for Caffe')
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO)
  ###########################
  convert(args.input, args.output)
