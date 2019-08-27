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

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

import sys
import os
#sys.path.insert(0, os.path.join(os.environ["CAFFE_ROOT"], "python"))
sys.path.append(os.path.join("/home/labuser/ML/tools/caffe1/", "python"))

import caffe
import numpy as np
import argparse
import logging

def convert(ckfile_path, cmodel_path, out_path):
  #
  cnet = caffe.Net(cmodel_path, caffe.TEST)
  #
  reader = pywrap_tensorflow.NewCheckpointReader(ckfile_path)
  var_to_shape_map = reader.get_variable_to_shape_map()
  met_fullconnect = False
  ddim = cnet.blobs[cnet.inputs[0]].data.shape[1:]
  last_act_ch = ddim[0]
  for key in sorted(var_to_shape_map):
    logging.debug("=========================================")
    logging.debug('key: ' + key)
    key_items = key.rsplit('/',1)
    logging.debug('key_items: ' + " ".join(key_items))
    lname = key_items[0]
    logging.debug('lname: ' + lname)
    if lname in ['train', 'cross_entropy', 'global_step', 'total_loss', 'beta1_power', 'beta2_power', 'iou']:
        logging.info("skip tensor, {}".format(key))
        continue
    wtype = key_items[1]
    logging.debug('wtype: ' + wtype)
    if wtype in ['ExponentialMovingAverage', 'Adam', 'Adam_1']:
        logging.info("skip tensor, {}".format(key))
        continue
    #----
    logging.info("tensor_name: "+key)
    if True:
        if lname.endswith('/bn'):
            lname = lname[:-3]
	    print('new lname: ', lname)
        if wtype in ['Momentum', 'biased', 'local_step']:
            logging.info("skip tensor, {}".format(key))
            continue
    if False:
        _lname = lname.split('/')[0]
        if _lname in ['bn']:
            logging.info("skip tensor, {}".format(key))
            continue
        #if 'moving_mean' in lname or 'moving_variance' in lname:
        #    logging.info("skip tensor, {}".format(key))
        #    continue
    #----
    if 'weight_loss' in lname:
        logging.info("skip tensor, {}".format(key))
        continue
    try:
        lidx = list(cnet._layer_names).index(lname)
    except:
        assert False, "not found, {}".format(lname)
    layer = cnet.layers[lidx]
    logging.debug('layer:'+layer.type)
    if layer.type in ['Convolution', 'InnerProduct', 'BinaryConvolution', 'BinaryInnerProduct']:
        if len(key_items) == 2:
            values = reader.get_tensor(key)
            logging.info("    tf_dim: {}".format(','.join(map(str,values.shape))))    # dim of tensor
            #if wtype == 'weights':
            if wtype == 'weights' or wtype == 'depth_filter' or wtype == 'kernels':
                logging.info("    caffe_dim: {}".format(','.join(map(str,cnet.params[lname][0].data.shape))))     # dim of caffe kernel
                dimaxis = len(cnet.params[lname][0].data.shape)
                if dimaxis == 4:    # Convolution
                    assert 'Convolution' in layer.type
                    if len(values.shape) == 4:
                        # (H,W,C,N) -> (N,C,H,W)
                        cnet.params[lname][0].data[...] = values.transpose(3,2,0,1)
                    elif len(values.shape) == 3:
                        # (H,W,N) -> (N,1,H,W)
                        cnet.params[lname][0].data[...] = np.expand_dims(values.transpose(2,0,1), 1)
                    else:
                        assert False
                    last_act_ch = layer.blobs[0].num
		    logging.debug('processed')
                elif dimaxis == 2:  # InnerProduct
                    assert 'InnerProduct' in layer.type
                    if not met_fullconnect:
                        logging.info("    (H,W,{0}) to ({0},H,W)".format(last_act_ch))
                        chsize = layer.blobs[0].channels // last_act_ch
                        values = values.reshape(chsize, last_act_ch, layer.blobs[0].num).transpose(1,0,2).reshape(layer.blobs[0].channels, layer.blobs[0].num)
                        met_fullconnect = True
                    cnet.params[lname][0].data[...] = values.transpose(1,0)
		    logging.debug('processed')
                else:
                    raise InternalError
            elif wtype == 'biases' or wtype == 'bias':
                logging.info("    caffe_dim: {}".format(','.join(map(str,cnet.params[lname][1].data.shape))))     # dim of caffe bias
                cnet.params[lname][1].data[...] = values
		logging.debug('processed')
            else:
                logging.warning("skip unknown type, {}".format(wtype))
                pass
        elif len(key_items) == 3:
            if key_items[1] == 'batchnorm':
                values = reader.get_tensor(key)
                #logging.debug(values.shape)
                lname = '/'.join(key_items[0:2])
                if key_items[2] in ['alpha', 'beta', 'gamma']:
                    lname += '/scale'
                #logging.debug(lname)
                if key_items[2] == 'mean' or key_item[2] == 'moving_mean':
                    cnet.params[lname][0].data[...] = values
                    cnet.params[lname][2].data[...] = 1.0
		    logging.debug('processed')
                elif key_items[2] == 'var' or key_item[2] == 'moving_variance':
                    cnet.params[lname][1].data[...] = values
		    logging.debug('processed')
                elif key_items[2] == 'alpha' or key_items[2] == 'gamma':
                    cnet.params[lname][0].data[...] = values
		    logging.debug('processed')
                elif key_items[2] == 'beta':
                    cnet.params[lname][1].data[...] = values
		    logging.debug('processed')
                else:
                    logging.warning("{}".format(layer.type))
                    pass
            else:
                logging.warning("{}".format(layer.type))
                pass
        else:
            logging.warning("{}".format(layer.type))
            pass
    elif layer.type == 'BatchNorm':
        values = reader.get_tensor(key)
        #logging.debug(values.shape)
        lname = '/'.join(key_items[:-1])
        logging.debug(lname)
        if key_items[-1] in ['alpha', 'beta', 'gamma']:
            #lname = '/'.join(key_items[:-1] + ['scale'])
            lname += '/scale'
        logging.debug(lname)
        if key_items[-1] == 'mean' or key_items[-1] == 'moving_mean':
            cnet.params[lname][0].data[...] = values
            cnet.params[lname][2].data[...] = 1.0
	    logging.debug('processed')
        elif key_items[-1] == 'var' or key_items[-1] == 'moving_variance':
            cnet.params[lname][1].data[...] = values
	    logging.debug('processed')
        elif key_items[-1] == 'alpha' or key_items[-1] == 'gamma':
            cnet.params[lname][0].data[...] = values
	    logging.debug('processed')
        elif key_items[-1] == 'beta':
            cnet.params[lname][1].data[...] = values
	    logging.debug('processed')
        else:
            logging.warning("{}".format(layer.type))
            pass
    else:
        logging.error("{}".format(layer.type))
        assert(False)
    cnet.save(out_path)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Tensorflow to Caffe Network')
  parser.add_argument('--ckpt', type=str, default="test.ckpt", help='checkpoint generated from TF')
  parser.add_argument('--net', type=str, default="test.proto", help='prototxt for Caffe')
  parser.add_argument('--output', type=str, default="test.caffemodel", help='caffemodel for Caffe')
  args = parser.parse_args()
  logging.basicConfig(level=logging.DEBUG)
  ###########################
  convert(args.ckpt, args.net, args.output)
