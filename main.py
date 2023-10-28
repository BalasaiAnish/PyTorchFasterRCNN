#Importing custom functions
from utils import *
#Importing required libraries
import torch

import torchvision
#Importing transforms from torchvision
from torchvision.transforms import v2
#Importing a pre trained model
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
#Importing a function to draw bounding boxes
from torchvision.utils import draw_bounding_boxes

#Importing OpenCV to capture and process images from the webcam
import cv2

#Setting device as gpu if applicable to speed up the model
device='cuda' if torch.cuda.is_available() else 'cpu'

#Loading the model on the latest version of weights
weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model=fasterrcnn_resnet50_fpn(weights=weights)

#Setting model to eval mode and moving it to device to speed up computations
model.eval()
model.to(device)

#Opening the webcam to start taking video
cap=cv2.VideoCapture(0)

#Infinite loop to keep running unitl the esc key is pressed
while True:
  #Takes an image from the wecam and return a tensor suitable as an inout for the model
	image=takePicture(cap,size=(224,224))

  #Running the image throught the neural network
	output=model(image)[0]

  #Take the image and output of the model as inputs and return an image with bounding boxes and labels drawn on it
	output_image=drawBoundingBoxes(image,output)
  
  #Destroying previously created window
	cv2.destroyAllWindows()

  #Displaying new image with labels and bounding boxes
	cv2.imshow('out',output_image)

  #Exits the loop if esc key is pressed
  if cv2.waitKey(100)==27:
		break

  
