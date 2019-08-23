# tensorflow_slim
train/test experience with slim packages in tensorflow

# Installation
## tensorflow 
- For CPU:
pip install tensorflow
- For GPU:
pip install tensorflow-gpu

## install slim's "nets" package by clone tensorflow/models
git clone https://github.com/tensorflow/models
cd models/research/slim/
python setup.py install

# Prepare data with tfrecord format
cd models/research/slim/
assume that your dataset name is *custom*

- Step 1: Modify dataset_dir, datatype, output_dir to create *tfrecord* format for data in scripts "convert_custom_dataset_2tfrecord.py"
- Step 2: Add a file "custom.py" in folder datasets that like "flowers.py"
- Step 3: Add some lines in "datasets/dataset_factory.py": "from datasets import custom" and for datasets_map "'custom': custom,"

