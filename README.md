# Object detection in an Urban Environment

## [Project Writeup](project_writeup.md)

## Data

We use data from the [Waymo Open dataset](https://waymo.com/open/). The files can be downloaded directly from the website as tar files or from the [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/) as individual tf records. 

## Structure

The core files in this repository will be organized as follows:
```
- exploratory_data_analysis.py: to confirm correct bounding box in images and analyze the dataset 
- explore_augmentation.py: to test augmentations in the dataset
- create_splits.py: to split data into training, validation and testing
- edit_config.py: to create a new configuration for training
- model_main_tf2.py: to launch training
- exporter_main_v2.py: to create an inference model
- inference_video.py: to make a video of object detection results
```

Jupyter Notebook:
```
Exploratory Data Analysis.ipynb
Explore augmentations.ipynb
```

The experiments folder are organized as follow:
```
experiments/
    - experiment0/... (initial pipeline_config)
    - experiment1/... (modified pipeline_config for improvements)
    - experiment2/... (modified pipeline_config for improvements)
```

The data folder contains:
```
data/
    - processed_data: contain the processed data from Waymo Open dataset (empty to start)
    - test: contain the test data (empty to start)
    - train: contain the train data (empty to start)
    - val: contain the val data (empty to start)
```

The training folder contains:
```
training/
    - pre-trained model (contains the checkpoints of the pretrained models. see the instruction in "Edit the config file" section)
    - reference (empty to start)
```

## Prerequisites

### Local Setup (Nvidia GPU)

Use the provided Dockerfile and requirements in the [build directory](./build).

Follow [the README](./build/README.md) to create a docker container and install all prerequisites.

An alternative way is to download each library individually as well as [python library dependencies](./build/requirements.txt). The following setup in the local machine can run the program successfully:

* Ubuntu 20.04
* Python 3.8.10
* CUDA 11.2
* cuDNN 8.1
* TensorFlow 2.6

Note that in `model_lib_v2.py` (TensorFlow Object Detection API), put `from object_detection import eval_util` in the last import to avoid the potential segmentation fault.

## Instructions

1. Clone this repo: `git clone https://github.com/PoChang007/Object_Detection_in_Urban_Env.git`
2. `cd Object_Detection_in_Urban_Env`

### Download and process the data

Process the downloaded [Waymo Open dataset](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/) by using `process_waymo_data.py`.
```
python3 process_waymo_data.py --src_dir {temp_dir_for_raw_files} --dst_dir data/processed_data
```

Or use `download_process.py` to download and process Waymo Open dataset, 
```
python3 download_process.py --data_dir data/processed_data --temp_dir {temp_dir_for_raw_files}
```

### Create the splits

Execute the script `create_splits.py` to split the dataset into training, validation and testing.
```
python3 create_splits.py --data_dir data/processed_data
```

### Edit the config file

The config that we will use for this project is `pipeline.config`, which is the config for a SSD Resnet 50 640x640 model.

First, download the [pretrained model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) and move it to `training/pretrained-models/`. 

Now we need to edit the config files to change the location of the training and validation files, as well as the location of the label_map file, pretrained weights. We also need to adjust the batch size. To do so, run the following:
```
python3 edit_config.py --train_dir data/train/ --eval_dir data/val/ --batch_size 4 --checkpoint ./training/pretrained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map label_map.pbtxt
```
A new config file has been created, `pipeline_new.config`.

### Training

Launch an experiment with the Tensorflow object detection API. Create a folder `training/reference`. Move the `pipeline_new.config` to this folder. We will now have to launch two processes: 
* a training process:
```
python3 model_main_tf2.py --model_dir=training/reference/ --pipeline_config_path=training/reference/pipeline_new.config
```
* an evaluation process:
```
python3 model_main_tf2.py --model_dir=training/reference/ --pipeline_config_path=training/reference/pipeline_new.config --checkpoint_dir=training/reference/
```

NOTE: both processes will display some Tensorflow warnings.

To monitor the training, launch a tensorboard instance by running `tensorboard --logdir=training`.

### Improve the performances

The initial experiment may not yield optimal results. However, we can make multiple changes to the config file to improve this model, for example, do the data augmentation. The [`preprocessor.proto`](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto) file contains the different data augmentation method available in the Tf Object Detection API. 

### Export the trained model

```
python3 ./exporter_main_v2.py --input_type image_tensor --pipeline_config_path training/reference/pipeline_new.config --trained_checkpoint_dir training/reference --output_directory training/experiment0/exported_model/
```

### Creating an animation

```
python3 inference_video.py --labelmap_path label_map.pbtxt --model_path training/experiment0/exported_model/saved_model --tf_record_path data/test/segment-tripID.tfrecord --config_path training/experiment0/exported_model/pipeline.config --output_path animation.mp4
```