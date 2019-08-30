import os

output_dir='../../../outputs'
dataset_name='peta_v2'
#dataset_name='getty_dataset_02'
dataset_dir='../../../data/'+dataset_name
model_name='resnet_v1_50' #inception_v3  #mobilenet_v1 #resnet_v1_50

#01.training
## hyper-params
num_thread=8
input_size=224
batch_size=32
quant_delay=10
save_ckpt_every_seconds=300
log_every_n_steps=100
maximum_steps=600000
ignore_missing=True
lr=0.0001
end_lr=0.0001
lr_decay_factor=0.94
optimizer='rmsprop'
## others
checkpoint_train_path='../../../outputs/mobilenet_v1_224_getty_dataset_02/qt_model/model.ckpt-12152'  #can be directory or specific checkpoint to fine-tune
checkpoint_train_path='/home/atsg/PycharmProjects/gvh205_py3/tensorflow_slim/models/downloaded/resnet50_v1/resnet_v1_50.ckpt' #training from scratch
checkpoint_train_path='' #training from scratch
if(checkpoint_train_path !=''):
    lr=0.001
checkpoint_exclude= 'resnet_v1_50/logits,resnet_v1_50/AuxLogits'# 'resnet_v1_50/logits,resnet_v1_50/AuxLogits'  #'InceptionV3/Logits,InceptionV3/AuxLogits'  #'MobilenetV1/Logits'
trainable_scopes='' #'InceptionV3/Logits,InceptionV3/AuxLogits' #'MobilenetV1/Logits'

#02.evaluate
checkpoint_eval_path='../../../models/downloaded/resnet50_v1'  #can be directory or specific checkpoint
log_dir=os.path.dirname(checkpoint_eval_path)
cpu=True

#03.export inference graph
type='qt'
checkpoint_path ='../../../models/downloaded/resnet50_v1/model.ckpt-3443'
output_graph = os.path.join('../../../outputs/tflite', model_name+'_'+str(input_size)+ '_'+type +'_graph.pb')
output_file = os.path.join('../../../outputs/tflite', model_name+'_'+str(input_size)+ '_'+type +'_frozen.pb')

#04.freeze_graph
output_node='resnet_v1_50/predictions/Softmax'

#05.convert to tflite
input_node='input'
output_tflite=os.path.join('../../../outputs/tflite', model_name+'_'+str(input_size)+ '_'+type +'.tflite')
