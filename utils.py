#Importing necessary libraries
import torch

import torchvision
#Importing torchvision transforms to trnsform taken images
from torchvision.transforms import v2

#Importing OpenCV to take images from the webcam, perform processing on them, and display them
import cv2

#Function that takes the video capture object from VideoCapture and an image size and returns a tensor that can be fed into the model
def takePicture(cap,size=(224,224),device):
	success,image=cap.read()
	image=cv2.resize(image,size)
	image=torch.from_numpy(image)
	image=torch.permute(image,(2,0,1))
	image=torch.unsqueeze(image,dim=1)
	image=v2.ToDtype(torch.float32) (image)
	image=image.to(device)
	return image

#Fucntion that takes the image tensor and output and returns a numpy array image that OpenCV can display with labels and bounding boxes
def drawBoundingBoxes(image,output):
	image=torch.squeeze(image,1)
	image=v2.ToDtype(torch.uint8) (image)
	boxes=output['boxes']
	labels=[]
	for label in output['labels']:
		labels.append(weights.meta["categories"][label])
	
	output_image=draw_bounding_boxes(image=image,boxes=boxes,labels=labels,width=5)
	
	output_image=torch.permute(output_image,(1,2,0))
	output_image=output_image.cpu().detach().numpy()
	return output_image
