IN=cmd_seven/checkpoint/tinyex4_conv.pbtxt
OUT=cmd_seven/test.proto

python tf2caffe_net.py --input $IN --output $OUT
