#Importing necessary libraries
import torch

import torchvision

#Importing torchvision transforms to trnsform taken images
from torchvision.transforms import v2

#Importing draw_bounding_boxes
from torchvision.utils import draw_bounding_boxes

#Importing numpy for processing output of the model
import numpy as np

#Importing OpenCV to take images from the webcam, perform processing on them, and display them
import cv2

#Function that takes the video capture object from VideoCapture and an image size and returns a tensor that can be fed into the model
#Function that takes the video capture object from VideoCapture and an image size and returns a tensor that can be fed into the model
def takePicture(cap, device, size=(224,224)):
	success,image=cap.read()
	image = cv2.resize(image,size)
	image = torch.from_numpy(image)
	image = torch.permute(image,(2,0,1))
	image = torch.unsqueeze(image,dim=1)
	image = v2.ToDtype(torch.float32) (image)
	image = image.to(device)
	return image

#Fucntion that takes the image tensor and output and returns a numpy array image that OpenCV can display with labels and bounding boxes
def drawBoundingBoxes(image, output, categories, threshold=0.5):
	#Making image tensor compatible with draw_bounding_boxes function
	image = torch.squeeze(image,1)
	image = v2.ToDtype(torch.uint8) (image)
	
	#Converting output tensors of the model to numpy arrays for processing
	initial_boxes  = output['boxes'].cpu().detach().numpy()
	initial_labels = output['labels'].cpu().detach().numpy()
	initial_scores = output['scores'].cpu().detach().numpy()

	#Initialsing list for box coordinates
	boxes = []
	
	#Initialising lists for labels and corresponding scores
	labels = []
	scores = []

	#Iterating over all detections
	for i in range(len(initial_scores)):	
	
		#Only appending those detections with a higher probablility than threshold
		if initial_scores[i] > threshold:
			boxes.append(initial_boxes[i])
			scores.append(initial_scores[i])
			labels.append(categories[initial_labels[i]])
	
	#Adding labels and scores together
	for j in range(len(labels)):
		labels[j] = labels[j]+', '+str(round(scores[j],2))
		
	#Converting boxes array to a numpy array and then to a tensor	
	boxes = np.array(boxes)
	boxes = torch.from_numpy(boxes)
		
	#Drawing bounding boxes on the output image	
	output_image = draw_bounding_boxes(image=image, boxes=boxes, labels=labels,width=3)
	
	#Converting the output image to numpy array suitable for OpenCV
	output_image = torch.permute(output_image, (1,2,0))
	output_image = output_image.cpu().detach().numpy()
	return output_image
