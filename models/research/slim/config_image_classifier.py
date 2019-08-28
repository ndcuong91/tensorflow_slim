import os

train_log_dir='../../../train_logs'
dataset_name='getty_dataset_02'
dataset_dir='../../../data/'+dataset_name
model_name='mobilenet_v1' #inception_v3  #mobilenet_v1

#training
## hyper-params
num_thread=8
input_size=224
batch_size=32
quant_delay=0
save_ckpt_every_seconds=300
log_every_n_steps=100
maximum_steps=600000
ignore_missing=True
lr=0.0001
end_lr=0.0001
lr_decay_factor=0.94
optimizer='rmsprop'
## others
checkpoint_train_path='../../../train_logs/mobilenet_v1_224_getty_dataset_02/qt_model/model.ckpt-12152'  #can be directory or specific checkpoint to fine-tune
#checkpoint_train_path='' #training from scratch
if(checkpoint_train_path !=''):
    lr=0.001
checkpoint_exclude= ''  #'InceptionV3/Logits,InceptionV3/AuxLogits'  #'MobilenetV1/Logits'
trainable_scopes='' #'InceptionV3/Logits,InceptionV3/AuxLogits' #'MobilenetV1/Logits'

#evaluate
checkpoint_eval_path='../../../train_logs/mobilenet_v1_224_getty_dataset_02/qt_model/model.ckpt-23643'  #can be directory or specific checkpoint
log_dir=os.path.dirname(checkpoint_eval_path)

#export inference graph
output_graph='../../../outputs/mobilenet_qt_v1_224_9000.pb'
checkpoint='../../../outputs/2019-08-23_18.06_quant_500000_steps_90_from_scratch/model.ckpt-500000'
output_file='../../../outputs/mobilenet_v1_qt_224_9000_with_ckpt.pb'

