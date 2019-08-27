#!/bin/env python
#
#  T+ CNN ML program generator
#    for Keyword Spotting

CheckAct = False

import numpy as np
import logging
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.environ['CAFFE_ROOT'], "python"))
import caffe
import wave

tool_root = os.path.join(os.environ['HOME'], 'ML', 'mldevkit_tplus')
sys.path.append(os.path.join(tool_root, 'nnfpga', 'fwgen'))
import cnnT_layer as layer
from cnn_util import ceildiv, dump_mem, dump_word

import cnnT_hw as hw
import cnnT_hw_prog as hw_prog
hw.simEnable = True

if hw.simEnable and CheckAct:
    sys.path.append(os.path.join(tool_root, 'nnfpga', 'layer'))
    from convolution_layer import do_convolution_fx
    from batchnorm_layer import do_batchnorm_fx
    from fullconnect_layer import do_fullconnect_fx
    from relu_layer import do_relu_hw
    from pooling_layer import do_pooling

min16_val = (1<<15)*-1
max16_val = (1<<15)-1

# Fraction of layers
cfrac = 1024
dfracbits = {
    'fingerprint':   10,
    'Convolution2':  7,
    'Convolution3':  6,
    'Convolution4':  7,
    'InnerProduct1': 9,
}
dfracs = dict([(k, 2**v) for k,v in dfracbits.iteritems()])

# Network parameters           size
data_dim   = (2, 32, 32)    # 0x800
pool1_dim  = (8, 16, 16)    # 0x800
pool2_dim  = (8,  8, 8)     # 0x200
pool3_dim  = (8,  4, 4)     # 0x080
output_dim = (3,)

# SPRAM size: 16k entries(0x4000)
INPUT_ADDR  = 0x0
POOL1_ADDR  = 0x800
POOL2_ADDR  = 0x000
POOL3_ADDR  = 0x800
OUTPUT_ADDR = 0xF00

################################################################################
def run_sim(cnet, data):
    ### load input
    layer.create_hw()
    layer.write_data(data, INPUT_ADDR)
    ### stage 1
    logging.info('stage 1')
    hw_prog.put_comment('stage 1')
    scl_cfrac = cfrac * dfracs['Convolution2'] / dfracs['fingerprint'] 
    conv1_wt = np.rint(np.clip(cnet.params['conv1'][0].data * scl_cfrac, min16_val, max16_val)).astype(np.int32)
    scale_factor = 0. if cnet.params['conv1/batchnorm'][2].data[0] == 0. else (1./cnet.params['conv1/batchnorm'][2].data[0])
    bn1_mean = np.rint(np.clip(cnet.params['conv1/batchnorm'][0].data * scale_factor * dfracs['Convolution2'], min16_val, max16_val)).astype(np.int32)
    bn1_variance = cnet.params['conv1/batchnorm'][1].data * scale_factor
    bn1_scale = cnet.params['conv1/batchnorm/scale'][0].data * dfracs['Convolution3'] / dfracs['Convolution2']
    bn1_bias = np.rint(np.clip(cnet.params['conv1/batchnorm/scale'][1].data * dfracs['Convolution3'], min16_val, max16_val)).astype(np.int16)
    layer.run_conv2d_stage(conv1_wt, 
                           bn1_mean, bn1_variance, bn1_scale, bn1_bias,
                           data_dim, conv_pad=1,
                           inbase=INPUT_ADDR, outbase=POOL1_ADDR)
    if CheckAct:
        _conv = do_convolution_fx(conv1_wt, np.zeros((pool1_dim[0],), dtype=np.int16), data, pad=1, roundup=True)
        dump_mem("conv1_fx.txt", _conv)
        _bn = do_batchnorm_fx(bn1_mean, bn1_variance, bn1_scale, bn1_bias, _conv, roundup=True)
        _relu = do_relu_hw(_bn)
        refd = do_pooling(_relu, stride=2, kwidth=2, kheight=2, pad=0)
        _pool1 = layer.read_data(pool1_dim, POOL1_ADDR)
        assert np.sum(np.abs(refd - _pool1)) == 0
    if False:
        temp = layer.read_data(pool1_dim, POOL1_ADDR)
        dump_mem("pool1_{:x}.txt".format(POOL1_ADDR), temp)
    ### stage 2
    logging.info('stage 2')
    hw_prog.put_comment('stage 2')
    conv2_wt = np.rint(np.clip(cnet.params['conv2'][0].data * cfrac, min16_val, max16_val)).astype(np.int32)
    scale_factor = 0. if cnet.params['conv2/batchnorm'][2].data[0] == 0. else (1./cnet.params['conv2/batchnorm'][2].data[0])
    bn2_mean = np.rint(np.clip(cnet.params['conv2/batchnorm'][0].data * scale_factor * dfracs['Convolution3'], min16_val, max16_val)).astype(np.int32)
    bn2_variance = cnet.params['conv2/batchnorm'][1].data * scale_factor
    bn2_scale = cnet.params['conv2/batchnorm/scale'][0].data * dfracs['Convolution4'] / dfracs['Convolution3']
    bn2_bias = np.rint(np.clip(cnet.params['conv2/batchnorm/scale'][1].data * dfracs['Convolution4'], min16_val, max16_val)).astype(np.int16)
    layer.run_conv2d_stage(conv2_wt,
                           bn2_mean, bn2_variance, bn2_scale, bn2_bias,
                           pool1_dim, conv_pad=1,
                           inbase=POOL1_ADDR, outbase=POOL2_ADDR)
    if CheckAct:
        _conv = do_convolution_fx(conv2_wt, np.zeros((pool2_dim[0],), dtype=np.int16), _pool1, pad=1, roundup=True)
        _bn = do_batchnorm_fx(bn2_mean, bn2_variance, bn2_scale, bn2_bias, _conv, roundup=True)
        _relu = do_relu_hw(_bn)
        refd = do_pooling(_relu, stride=2, kwidth=2, kheight=2, pad=0)
        _pool2 = layer.read_data(pool2_dim, POOL2_ADDR)
        assert np.sum(np.abs(refd - _pool2)) == 0
    if False:
        temp = layer.read_data(pool2_dim, POOL2_ADDR)
        dump_mem("pool2_{:x}.txt".format(POOL2_ADDR), temp)
    ### stage 3
    logging.info('stage 3')
    hw_prog.put_comment('stage 3')
    conv3_wt = np.rint(np.clip(cnet.params['conv3'][0].data * cfrac, min16_val, max16_val)).astype(np.int32)
    scale_factor = 0. if cnet.params['conv3/batchnorm'][2].data[0] == 0. else (1./cnet.params['conv3/batchnorm'][2].data[0])
    bn3_mean = np.rint(np.clip(cnet.params['conv3/batchnorm'][0].data * scale_factor * dfracs['Convolution4'], min16_val, max16_val)).astype(np.int32)
    bn3_variance = cnet.params['conv3/batchnorm'][1].data * scale_factor
    bn3_scale = cnet.params['conv3/batchnorm/scale'][0].data * dfracs['InnerProduct1'] / dfracs['Convolution4']
    bn3_bias = np.rint(np.clip(cnet.params['conv3/batchnorm/scale'][1].data * dfracs['InnerProduct1'], min16_val, max16_val)).astype(np.int16)
    layer.run_conv2d_stage(conv3_wt,
                           bn3_mean, bn3_variance, bn3_scale, bn3_bias,
                           pool2_dim, conv_pad=1,
                           inbase=POOL2_ADDR, outbase=POOL3_ADDR)
    if CheckAct:
        _conv = do_convolution_fx(conv3_wt, np.zeros((conv3_dim[0],), dtype=np.int16), _pool2, pad=1, roundup=True)
        _bn = do_batchnorm_fx(bn3_mean, bn3_variance, bn3_scale, bn3_bias, _conv, roundup=True)
        _relu = do_relu_hw(_bn)
        refd = do_pooling(_relu, stride=2, kwidth=2, kheight=2, pad=0)
        _pool3 = layer.read_data(pool3_dim, POOL3_ADDR)
        assert np.sum(np.abs(refd - _pool3)) == 0
    if False:
        temp = layer.read_data(pool3_dim, POOL3_ADDR)
        dump_mem("conv3_{:x}.txt".format(POOL3_ADDR), temp)
    ### stage 4
    logging.info('stage 4')
    hw_prog.put_comment('stage 4')
    fc4_wt = np.rint(np.clip(cnet.params['fc4'][0].data * cfrac, min16_val, max16_val)).astype(np.int32)
    fc4_bias = np.rint(np.clip(cnet.params['fc4'][1].data * dfracs['InnerProduct1'], min16_val, max16_val)).astype(np.int16)
    layer.run_fc_layer(fc4_wt, fc4_bias, pool3_dim, inbase=POOL3_ADDR, outbase=OUTPUT_ADDR)
    if CheckAct:
        refd = do_fullconnect_fx(fc4_wt, fc4_bias, _pool3, roundup=True)
        _fc4 = layer.read_data(output_dim, OUTPUT_ADDR)
        assert np.sum(np.abs(refd - _fc4)) == 0

    layer.read_result(OUTPUT_ADDR, output_dim[0])
    layer.end_engine()

    return layer.read_data(output_dim, OUTPUT_ADDR)


################################################################################
if __name__ == "__main__":
    #---------------------------------------
    # arg
    parser = argparse.ArgumentParser(description='FPGA program generator')
    parser.add_argument('--network', type=str, default="../import/scmddet.proto", help='Network in prototxt')
    parser.add_argument('--caffemodel', type=str, default="../import/scmddet.caffemodel", help='Caffemodel')
    parser.add_argument('--output', type=str, default="", help='Output')
    parser.add_argument('--input_wav', type=str, default="/data/speech_dataset/seven/fd395b74_nohash_2.wav", help='Input WAV file')
    args = parser.parse_args()
    #---------------------------------------
    # log
    logging.basicConfig(level=logging.INFO)
    log1 = logging.getLogger('cnnT_layer')
    log2 = logging.getLogger('cnnT_hw')
    #---------------------------------------
    # program output
    output_file = args.output.strip()
    hw_prog.outEnable = bool(output_file)
    if output_file:
        hw_prog.outFile = open(output_file,"w")
        hw_prog.outBinFile = open(output_file.rsplit('.',1)[0]+'.bin',"wb")
    #---------------------------------------
    # data input
    duration = 1.04
    samplerate = 16000
    total_samples = int(duration*samplerate)
    waveFile = wave.open(args.input_wav, 'rb')
    wav_data = waveFile.readframes(int(duration*samplerate))
    wav_data += b'\0'*1280     # zero padding
    #wav_data += wav_data
    audio_stream = np.fromstring(wav_data, dtype=np.int16)[:total_samples]
    print(audio_stream.shape)
    audio_data = audio_stream[::4].astype(np.float).reshape(1,1,-1,1)  # downsample
    audio_data /= (2**15)   # normalize
    audio_max = np.amax(audio_data)
    print("audio_data max: {}".format(audio_max))
    if audio_max <= 0.2:
        print("normalize volume")
        volup = 0.7 / audio_max
        audio_data *= volup
    #---------------------------------------
    # run Caffe
    cnet = caffe.Net(args.network, args.caffemodel, caffe.TEST)
    input_act = cnet.inputs[0]
    filter_act = 'fingerprint'
    cnet.blobs[input_act].data[...] = audio_data
    cnet.forward()
    filt_out = cnet.blobs[filter_act].data[0].squeeze()
    print("filter max: {}".format(np.amax(filt_out)))
    print("conv1: {} .. {}".format(np.amin(cnet.blobs['Convolution2'].data[0]), np.amax(cnet.blobs['Convolution2'].data[0])))
    #---------------------------------------
    # run
    data_frac = dfracs['fingerprint']
    data = np.rint(filt_out * data_frac).astype(np.int16)
    out = run_sim(cnet, data)
    logging.info(out)
    if out[2] > 0 and out[2] > out[0] and out[2] > out[1]:
        print("Detected")
    else:
        print("Not detected")
    if hw_prog.outEnable:
        dump_mem("in_000.txt", data)
        dump_mem("out_f00.txt", out)
        if True:
            assert len(cnet.params['freqconv']) == 1
            filt_w = np.rint(cnet.params['freqconv'][0].data * data_frac).astype(np.int16)
            dump_word('filter_w.txt', filt_w)
            dump_word('audio.txt', audio_stream)
            dump_word('fingerprint.txt', data)
