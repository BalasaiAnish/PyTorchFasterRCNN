#Object DEtection using transfer learning

Transfer learning is an extremely popular technique where a model that has already been trained extensively on a particular dataset is loaded with its weights and parameters intact for use in various applications. 

Here I have used the OpenCV library to take images from a camera device and perform some basic processing on them. I have also made use of PyTorch Deep learning framework to load the pre trained model and to transform the image taken in various ways.

The model being used here is Faster RCNN which is a two shot model that locates areas where an object might be and then performs image classifiction on those areas. This model return bounding boxes, labels, masks, and scores for each instance of a detected object which can be used to fine tune the final display.
