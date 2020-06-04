# ImageCaptioning
Image Captioning with LSTM Sequence to sequence model


# Python Library Requirements #
The project was developed in Google Colab and all of those libraries are already installed
* tensorflow
* keras
* open cv
* numpy
* matplotlib
* h5py
* pandas
* csv
* json
* urllib
* random
* pickle
* nltk

# Data Used - COCO caption dataset#
Downloading the dataset takes several hours and 20 GB
* train images - http://msvocds.blob.core.windows.net/coco2014/train2014.zip
* validation images - http://msvocds.blob.core.windows.net/coco2014/val2014.zip
* captions for both train and validation - http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip

It was hard to operate with such a big data in my google drive so 
I had to divide the whole dataset into several pieces of 100000 images. 


# Project structure

* dataset - folder previously had all the h5 files for parts of train images and also csv files f the names of the images. 
I deleted the image dataset after I go tall the feature vectors of all the images with InceptionV3 model

* embedding_dataset - folder where I stored all the image embeddings of datset parts

* annotations-2 - contains two json files containing the captions for train and validation sets

* checkpoints - contains all the checkpoints for all epoches of training and also the best epoch

* Embeddings_retrieval.ipynb - the Google Colab notebook which was used in order to 
get the feature vectors of all imaes in the dataset folder

* Image_Captioning_tf2.ipynb - the Google Colab notebook which was used in orer to train the model 
and save the checkpoints of the model in checkpoints folder.

* imgCap.py - the python file that contains the imageCaptioning class that was developed in the 
Image_Captioning_tf2.ipynb file

* Results.ipynb - the Google Colab to see the results of the imageCaptioning class and checkpoints

