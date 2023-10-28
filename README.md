# Object Detection using Transfer Learning and Faster RCNN

## Transfer Learning
Transfer Learning is a machine learning technique where a model that has already been trained is reused with its various weights and parameters intact to perform a task similar to the one it was trained to do. This helps save computational resources and time by eliminating the need to train models on the same data to perform similar tasks.

## Faster RCNN
Faster RCNN is a two shot object detection algorithm. The image is first scanned for the prescence of potential objects and then these potential objects undergo image classification to finally obtain the objects present in an image along with their bounding boxes, labels, masks, and probablitly scores. This particular model is implemented as outlined in this paper https://arxiv.org/abs/1506.01497. 

## PyTorch
PyTorch is a Deep Learning framework that has become increasingly poopular amongst researchers for its flexibility. Here I have used the built in model for Faster RCNN and a number of transformation functions to convert images to tensors, tensors to images, and perform pre preocessing on the images. PLease not that these transforms are part of the V2 transforms offered from Torchvision and are still in the Beta stage. This will provide a warning when running the program but it has not caused any errors and improves computational efficiency.

## OpenCV
OpenCV or Open Source Computer Vision is an open source library developed by the intel corporation. It has widespread use in both research and academia. Here it has been used to capture video from the webcam, perform pre processing on the taken images, and to display the output images with bounding boxes and labels.

## Uses 
This project was made to demonstrate how to perform transfer learning and apply it in the context of object recognition. For a varety of uses, pre trained models help save time, resources, and are more efficient in terms of computation. These models also don't sacrifice usability as they are extremely felxible and can be adpated to a wide variety of tasks with little to no changes. Completing this project helped me understand the ins and outs of object detecion, image processing, and the Faster RCNN algorithm.
