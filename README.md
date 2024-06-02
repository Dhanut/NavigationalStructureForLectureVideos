Based on this YouTube Video to setup the Env - https://www.youtube.com/watch?v=a1br6gW-8Ss
===============================================================================================================================================================================================================
Main Steps I have Followed :-

1. Script file
this contains:
 a.  PowerShell script to install and set up tensorflow object detection api. So download all the dependencies and set the env variables for the object detection.
 	01. Install Python
	02. Install C++ visual studio 
	03. Install visual studio build tools.
	04. Install pip
	05. Install tensorflow
	06. Install cv2
	07. Clone the model master GIT repo from Tensorflow (https://github.com/tensorflow/models/) where it has all the models and research work and convert it to python executable files.

Now can see the object detection folder is created with the clone in c directory.

 b. generate_labelmap.py:- to create labelmap file for object detection.
 c. generate_tfrecord.py :- to create tfrecord file for training and testing data.
 d. xml_to_csv.py :- Generate the csv file for training and testing images

2. Download labelImg tool for this link :- https://tzutalin.github.io/labelImg/

3. run this command for generating csv file for training and testing images 
 python xml_to_csv.py
4.  run this command for generting labelmap.pbtxt file 
 python generate_labelmap.py
5. Generate tfrecord file for training by this command 
 python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record
6. Generate tfrecord file for training by this command
 python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record

Download the state of art models from this site -> https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
Extract it in folder -> efficientdet_d0_coco17_tpu-32 folder will generate

Inside config folder can see the relevant config file. Copy and pate to images folder. After that do changes as below step.

7. Here are the argument to be updated on the config file for model training 
 num_classes: 5  [give number of classes here]
 learning_rate_base: 0.8e-3
 warmup_learning_rate: 0.0001
 fine_tune_checkpoint: "efficientdet_d0_coco17_tpu-32/checkpoint/ckpt-0"
 fine_tune_checkpoint_type: "detection"
 label_map_path: "images/labelmap.pbtxt"
 input_path: "train.record"
 label_map_path: "images/labelmap.pbtxt"
 input_path: "test.record"
 
8. train model command 
 python model_main_tf2.py --pipeline_config_path=training/ssd_efficientdet_d0_512x512_coco17_tpu-8.config --model_dir=training --alsologtostderr
 
9. export infrence graph command 
 python exporter_main_v2.py --trained_checkpoint_dir=training --pipeline_config_path=training/ssd_efficientdet_d0_512x512_coco17_tpu-8.config --output_directory inference_graph

===============================================================================================================================================================================================================
