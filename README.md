# Image_Captioning using VGG19 and Beam search
Image captioning model generates a textual description for a given input image.This repository contains the encoder-decoder framework based Image Captioning model.It requires both methods from computer vision to understand the content of the image and a language model from NLP to turn the understanding of the image into words in the right order. 
The encoder which is a VGG19 model pre-trained on ImageNet dataset extracts features from the input image and pass these features on to the decoder which is a Long short term memory (LSTM) network .Finally the decoder uses these features to output a natural language description.
# Libraries
```
os
pickle
keras
numpy
nltk
```
# Dataset
Used Flickr8k dataset since the size is 1GB. MS-COCO is 14GB
