# Nasutrad(clothers detection)
First we need to clone the Tensorflow models repository.

After we have cloned the repository we need to download Protobuf from this website.
Next we need to extract both of the folders using 7zip or winrar and while this is running we can install Cython, contextlib2, pillow, lxml, jupyter and matplotlib by typing:

pip install --user Cython 
pip install --user contextlib2 
pip install --user pillow 
pip install --user lxml 
pip install --user jupyter 
pip install --user matplotlib

hen we need to go into models/research and use protobuf to extract python files from the proto files in the object_detection/protos directory.
./bin/protoc object_detection/protos/*.proto --python_out=. 

But the * which stands for all files didn’t work for me so I wrote a little Python script to execute the command for each .proto file.

import os 
import sys 
args = sys.argv 
directory = args[1] 
protoc_path = args[2] 
for file in os.listdir(directory):
     if file.endswith(".proto"):
         os.system(protoc_path+" "+directory+"/"+file+" --python_out=.")
         
This file needs to be saved inside the research folder and I named it use_protobuf.py. Now we can use it by going into the console and typing:
Python use_protobuf.py  Example: python use_protobuf.py object_detection C:/Users/Gilbert/Downloads/bin/protoc 

Tensorflow Object Detection Tutorial #1 – Installation

In this tutorial we will install the Tensorflow Object Detection API and test it out using the object_detection_tutorial.ipynb file.

First we need to clone the Tensorflow models repository.

After we have cloned the repository we need to download Protobuf from this website.

Next we need to extract both of the folders using 7zip or winrar and while this is running we can install Cython, contextlib2, pillow, lxml, jupyter and matplotlib by typing:

1
2
3
4
5
6
pip install --user Cython 
pip install --user contextlib2 
pip install --user pillow 
pip install --user lxml 
pip install --user jupyter 
pip install --user matplotlib
Then we need to go into models/research and use protobuf to extract python files from the proto files in the object_detection/protos directory.

The official installation guide uses protobuf like:

./bin/protoc object_detection/protos/*.proto --python_out=. 
But the * which stands for all files didn’t work for me so I wrote a little Python script to execute the command for each .proto file.

1
2
3
4
5
6
7
8
import os 
import sys 
args = sys.argv 
directory = args[1] 
protoc_path = args[2] 
for file in os.listdir(directory):
     if file.endswith(".proto"):
         os.system(protoc_path+" "+directory+"/"+file+" --python_out=.")
This file needs to be saved inside the research folder and I named it use_protobuf.py. Now we can use it by going into the console and typing:

python use_protobuf.py  Example: python use_protobuf.py object_detection C:/Users/Gilbert/Downloads/bin/protoc 
Lastly we need to add the research and research slim folder to our enviroment variables and run the setup.py file.

On windows you need to at the path of the research folder and the research/slim to your PYTHONPATH enviroment variable (See Environment Setup) .

To run the setup.py file we need to navigate to tensorflow/models/research and run:
# From within TensorFlow/models/research/
python setup.py build
python setup.py install

his completes the installation of the object detection api. Now we can try it out by going into the object detection directory and typing jupyter notebook to open jupyter.

Then you can open the object_detection_tutorial.ipynb file and run all cells.

Creating your own object detector

Before we can get started creating the object detector we need data, which we can use for training.

To train a robust classifier, we need a lot of pictures which should differ a lot from each other. So they should have different backgrounds, random object and varying lighting conditions.

These images are pretty big because they have a high resolution so we want to transform them to a lower scale so the training process is faster.

I wrote a little script that makes it easy to transform the resolution of images.

from PIL import Image
import os
import argparse
 
def rescale_images(directory, size):
    for img in os.listdir(directory):
        im = Image.open(directory+img)
        im_resized = im.resize(size, Image.ANTIALIAS)
        im_resized.save(directory+img)
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rescale images")
    parser.add_argument('-d', '--directory', type=str, required=True, help='Directory containing the images')
    parser.add_argument('-s', '--size', type=int, nargs=2, required=True, metavar=('width', 'height'), help='Image size')
    args = parser.parse_args()
    rescale_images(args.directory, args.size)
    
    To use the script we need to save it in the parent directory of the images as something like transform_image_resolution.py and then go into the command line and type:
    python transform_image_resolution.py -d images/ -s 800 600
    
    Now that we have our images we need to move about 80 percent of the images into the object_detection/images/train directory and the other 20 percent in the object_detection/images/test directory.

In order to label our data we need some kind of image labeling software. LabelImg is a great tool for labeling images. It’s also freely available on Github and prebuilts can be downloaded easily.

https://tzutalin.github.io/labelImg/

After downloading and opening LabelImg you can open the training and testing directory using the “Open Dir” button.

To create the bounding box the “Create RectBox” button can be used. After creating the bounding box and annotating the image you need to click save. This process needs to be repeated for all images in the training and testing directory.

Generating TFRecords for training
With the images labeled, we need to create TFRecords that can be served as input data for training of the object detector. In order to create the TFRecords we will use two scripts from Dat Tran’s racoon detector(https://github.com/datitran/raccoon_dataset). Namly the xml_to_csv.py and generate_tfrecord.py files.

After downloading both scripts we can first of change the main method in the xml_to_csv file so we can transform the created xml files to csv correctly.

# Old:
def main():
    image_path = os.path.join(os.getcwd(), 'annotations')
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv('raccoon_labels.csv', index=None)
    print('Successfully converted xml to csv.')
# New:
def main():
    for folder in ['train', 'test']:
        image_path = os.path.join(os.getcwd(), ('images/' + folder))
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv(('images/'+folder+'_labels.csv'), index=None)
        print('Successfully converted xml to csv.')
       
Now we can transform our xml files to csvs by opening the command line and typing:
python xml_to_csv.py

These creates two file in the images directory. One called test_labels.csv and another one called train_labels.csv.

Before we can transform the new created files to TFRecords we need to change a few lines in the generate_tfrecords.py file.

Old:

# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'basketball':
        return 1
    elif row_label == 'shirt':
        return 2
    elif row_label == 'shoe':
        return 3
    else:
        return None
New:

def class_text_to_int(row_label):
    if row_label == 'Raspberry_Pi_3':
        return 1
    elif row_label == 'Arduino_Nano':
        return 2
    elif row_label == 'ESP8266':
        return 3
    elif row_label == 'Heltec_ESP32_Lora':
        return 4
    else:
        return None
If you are using a different dataset you need to replace the classnames with your own.

Now the TFRecords can be generated by typing:

python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record
python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record

These two commands generate a train.record and a test.record file which can be used to train our object detector.

Configuring training
The last thing we need to do before training is to create a label map and a training configuration file.

CREATE LABEL MAP
The label map maps a id to a name. We will but it in a folder called training, which is located in the object_detection directory. The labelmap for my detector can be seen below.

tem {
    id: 1
    name: 'shoes'
}

item {
    id: 2
    name: 'flipflops'
}

item {
    id: 3
    name: 'sandales'
}

item {
    id: 4
    name: 'boots'
}
item {
    id: 5
    name: 'socks'
}
item {
    id: 6
    name: 'sneakers'
}
The id number of each item should match the id of specified in the generate_tfrecord.py file.

CREATING TRAINING CONFIGURATION
Now we need to create a training configuration file. Because as my model of choice I will use faster_rcnn_inception, which just like a lot of other models can be downloaded from this page I will start with a sample config ( faster_rcnn_inception_v2_pets.config ), which can be found in the sample folder.

First of I will copy the file into the training folder and then I will open it using a text editor in order to change a few lines in the config.

Line 9: change the number of classes to number of object you want to detect (4 in my case)

Line 106: change fine_tune_checkpoint to the path of the model.ckpt file:

fine_tune_checkpoint: "C:/Users/Gilbert/Downloads/Other/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"
Line 123: change input_path to the path of the train.records file:

input_path: "C:/Users/Gilbert/Downloads/Other/models/research/object_detection/train.record"
Line 135: change input_path to the path of the test.records file:

input_path: "C:/Users/Gilbert/Downloads/Other/models/research/object_detection/test.record"
Line 125-137: change label_map_path to the path of the label map:

label_map_path: "C:/Users/Gilbert/Downloads/Other/models/research/object_detection/training/labelmap.pbtxt"
Line 130: change num_example to the number of images in your test folder.

To train the model we will use the train.py file, which is located in the object_detection/legacy folder. We will copy it into the object_detection folder and then we will open a command line and type:

python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/fast

If everything was setup correctly the training should begin shortly.

About every 5 minutes the current loss gets logged to Tensorboard. We can open Tensorboard by opening a second command line, navigating to the object_detection folder and typing:

tensorboard --logdir=training

Exporting inference graph
Now that we have a trained model we need to generate a inference graph, which can be used to run the model. For doing so we need to first of find out the highest saved step number. For this we need to navigate to the training directory and look for the model.ckpt file with the biggest index.

Then we can create the inference graph by typing the following command in the command line.

python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
XXXX represents the highest number.

Testing object detector

CHANGE object_detection_tutorial.ipynb file with the one in this GITHUB PROJECT 

Then type python -m notebook

and run the object_detection_tutorial.ipynb file from the server 
it should open the camera ready to detect clothes that you trained.

this project was done with the great help of these series of tutorials:
https://gilberttanner.com/2018/12/22/tensorflow-object-detection-tutorial-1-installation/
https://gilberttanner.com/2018/12/30/tensorflow-object-detection-tutorial-2-live-object-detection/
https://gilberttanner.com/2019/02/02/tensorflow-object-detection-tutorial-3-creating-your-own-object-detector/


