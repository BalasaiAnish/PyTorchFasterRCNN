
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

#Importing time for FPS calculation
import time

#Defining custom fucntions for modularity and clarity of code

#Function that takes the video capture object from VideoCapture and an image size and returns a tensor that can be fed into the model
def takePicture(cap,device,size=(224,224)):
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

#Setting device as gpu if applicable to speed up the model
device='cuda' if torch.cuda.is_available() else 'cpu'

#Loading the model on the latest version of weights
weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model=fasterrcnn_resnet50_fpn(weights=weights)

#Setting model to eval mode and moving it to device to speed up computations
model.eval()
model.to(device)

#Setting variables for FPS calclation
current_time=0
prev_time=0

#Opening the webcam to start taking video
cap=cv2.VideoCapture(0)

if not cap.isOpened:
	print('Unable to open wecam, please check your settings')
#Infinite loop to keep running unitl the esc key is pressed
while True:
	#Takes an image from the wecam and return a tensor suitable as an inout for the model
	image=takePicture(cap,device,size=(224,224))

	#Running the image throught the neural network
	output=model(image)[0]

	#Calculating FPS using 1/Processing_time formula
	current_time=time.time()
	fps=1/(current_time-prev_time)
	
	#Rounding off FPS to 2 decimals
	fps=round(fps,2)
	
	#Converting FPS to a string so it can be displayed
	fps=str(fps)
	fps="FPS: "+fps
	#Updating time values
	prev_time=current_time
	
	#Setting a font
	font=cv2.FONT_HERSHEY_SIMPLEX 

	#Take the image and output of the model as inputs and return an image with bounding boxes and labels drawn on it
	output_image=drawBoundingBoxes(image,output)
	
	#Placing the value for fps on to the screen
	cv2.putText(img=output_image, text=fps, org=(5,224-5), fontFace=font, fontScale=0.375, color=(0, 0, 255),thickness=1)

	#Displaying new image with labels and bounding boxes
	cv2.imshow('FasterRCNN',output_image)

	#Exits the loop if esc key is pressed
	if cv2.waitKey(1)==27:
		break

#Closing the webcam and destroying all created windows	
cap.release()
cv2.destroyAllWindows()

#Closing webcam and all created windows
cap.release()
cv2.destroyAllWindows() 
