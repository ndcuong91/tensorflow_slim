CKPT=cmd_seven/checkpoint/tinyex4_conv.ckpt-50000
NET=cmd_seven/scmddet.proto
OUT=cmd_seven/scmddet.caffemodel

python tf2caffe_weights.py --ckpt $CKPT --net $NET --output $OUT
