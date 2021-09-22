from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import cv2 as cv



import multiprocessing
import tensorflow.keras as tf
import pyttsx3
import math
import os
# use matplotlib if cv2.imshow() doesn't work
#import matplotlib.pyplot as plt

# C:\projects\TeachableMachineTest\Python
DIR_PATH = os.path.dirname(os.path.realpath(__file__))



def test():
	# Load the model
	print("--------------------------------")
	print("Loading model...")
	model = load_model('keras_model.h5', compile=False)
	#print(model)
	print("--------------------------------")

	#while True:
	# disable scientific notation for clarity
	np.set_printoptions(suppress=True)
	# Create the array of the right shape to feed into the keras model
	# The 'length' or number of images you can put into the array is
	# determined by the first position in the shape tuple, in this case 1.
	data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
	# Replace this with the path to your image
	#image = Image.open('image.png')
	#image = cv.VideoCapture(0)

	### capture image
	check, frame = cap.read()

	# crop to square for use with TM model
	#margin = int(((frameWidth-frameHeight)/2))
	square_frame = frame[0:frameHeight, margin:margin + frameHeight]
	# resize to 224x224 for use with TM model
	resized_img = cv.resize(square_frame, (224, 224))
	# convert image color to go to model
	model_img = cv.cvtColor(resized_img, cv.COLOR_BGR2RGB)







	#resize the image to a 224x224 with the same strategy as in TM2:
	#resizing the image to be at least 224x224 and then cropping from the center
	#size = (224, 224)
	#image = ImageOps.fit(image, size, Image.ANTIALIAS)

	#turn the image into a numpy array
	#image_array = np.asarray(image)
	### turn the image into a numpy array
	image_array = np.asarray(model_img)

	# Remove alpha channel
	#image_array = image_array[:,:,:3]
	#print(image_array)

	# Normalize the image
	normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
	# Load the image into the array
	#print(normalized_image_array)
	data[0] = normalized_image_array
	#print(data)
	# run the inference
	prediction = model.predict(data)
	#print("Prediction: ")
	#print(prediction)


	## confidence threshold is 90%.
	conf_threshold = 90
	confidence = []
	conf_label = ""
	threshold_class = ""

	# create blach border at bottom for labels
	per_line = 2  # number of classes per line of text
	bordered_frame = cv.copyMakeBorder(
		square_frame,
		top=0,
		bottom=30 + 15*math.ceil(len(classes)/per_line),
		left=0,
		right=0,
		borderType=cv.BORDER_CONSTANT,
		value=[0, 0, 0]
	)



def main():
	print("========== Main ==========")
	# read .txt file to get labels
	labels_path = f"{DIR_PATH}\\converted_keras\\labels.txt"

	# open input file label.txt
	labelsfile = open(labels_path, 'r', encoding='utf-8', errors='ignore')

	# initialize images classes and read in lines until there are no more
	classes = []
	line = labelsfile.readline()
	while line:
		# retrieve just class name and append to classes
		classes.append(line.split(' ', 1)[1].rstrip())
		line = labelsfile.readline()
	# close label file
	labelsfile.close()

	# load the teachable machine model
	model_path = f"{DIR_PATH}\\converted_keras\\keras_model.h5"
	model = tf.models.load_model(model_path, compile=False)

	# initialize webcam video object
	cap = cv.VideoCapture(0)

	# width & height of webcam video in pixels -> adjust to your size
    # adjust values if you see black bars on the sides of capture window
	frameWidth = 1024
	frameHeight = 768

	# set width and height in pixels
	cap.set(cv.CAP_PROP_FRAME_WIDTH, frameWidth)
	cap.set(cv.CAP_PROP_FRAME_HEIGHT, frameHeight)
    # enable auto gain
	cap.set(cv.CAP_PROP_GAIN, 0)

	# keeps program running forever until ctrl+c or window is closed
	while True:
		# disable scientific notation for clarity
		np.set_printoptions(suppress=True)

        # Create the array of the right shape to feed into the keras model.
        # We are inputting 1x 224x224 pixel RGB image.
		data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

		# capture image
		check, frame = cap.read()
        # mirror image - mirrored by default in Teachable Machine
        # depending upon your computer/webcam, you may have to flip the video
        # frame = cv2.flip(frame, 1)

        # crop to square for use with TM model
		margin = int(((frameWidth-frameHeight)/2))
		square_frame = frame[0:frameHeight, margin:margin + frameHeight]
        # resize to 224x224 for use with TM model
		resized_img = cv.resize(square_frame, (224, 224))
        # convert image color to go to model
		model_img = cv.cvtColor(resized_img, cv.COLOR_BGR2RGB)

		# turn the image into a numpy array
		image_array = np.asarray(model_img)
        # normalize the image
		normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        # load the image into the array
		data[0] = normalized_image_array

        # run the prediction
		predictions = model.predict(data)

		# confidence threshold is 90%.
		conf_threshold = 90
		confidence = []
		conf_label = ""
		threshold_class = ""

		# create blach border at bottom for labels (BGR) 
		per_line = 2  # number of classes per line of text
		bordered_frame = cv.copyMakeBorder(
			square_frame,
			top=0,
			bottom=30 + 15*math.ceil(len(classes)/per_line),
			left=0,
			right=0,
			borderType=cv.BORDER_CONSTANT,
			value=[0, 0, 0]
		)

		# Window name in which image is displayed
		window_name = 'Capturing'
		  
		# font
		font = cv.FONT_HERSHEY_SIMPLEX
		  
		# org
		org = (20, 565)
		org2 = (20, 590)
		  
		# fontScale
		fontScale = 0.5
		   
		# White color in BGR
		color = (255, 255, 255)
		  
		# Line thickness of 2 px
		thickness = 1
		   
		# Using cv2.putText() method
		# = cv.putText(bordered_frame, "Linha 1", org, font, fontScale, color, thickness, cv.LINE_AA)
		#test_frame = cv.putText(bordered_frame, "Linha 2", org2, font, fontScale, color, thickness, cv.LINE_AA)



		# for each one of the classes
		for i in range(0, len(classes)):
            # scale prediction confidence to % and apppend to 1-D list
			confidence.append(int(predictions[0][i]*100))
            # put text per line based on number of classes per line
			if (i != 0 and not i % per_line):
				print(confidence)
				# org=(int(0), int(frameHeight+25+15*math.ceil(i/per_line))),
				cv.putText(
					img=bordered_frame,
					text=conf_label,
					org=(20, 565),
					fontFace=cv.FONT_HERSHEY_SIMPLEX,
					fontScale=0.5,
					color=(255, 255, 255)
				)
				conf_label = ""
            # append classes and confidences to text for label
			conf_label += classes[i] + ": " + str(confidence[i]) + "%; "
            # prints last line ---  org=(int(0), int(frameHeight+25+15*math.ceil((i+1)/per_line))),
			if (i == (len(classes)-1)):
				cv.putText(
					img=bordered_frame,
					text=conf_label,
					org=(20, 590),
					fontFace=cv.FONT_HERSHEY_SIMPLEX,
					fontScale=0.5,
					color=(255, 255, 255)
				)
				conf_label = ""
            # if above confidence threshold, send to queue
			if confidence[i] > conf_threshold:
				##speakQ.put(classes[i])
				threshold_class = classes[i]
        # add label class above confidence threshold ---- org=(int(0), int(frameHeight+20)),
		cv.putText(
			img=bordered_frame,
			text=threshold_class,
			org=(400, 580),
			fontFace=cv.FONT_HERSHEY_SIMPLEX,
			fontScale=0.75,
			color=(255, 255, 255)
		)

		# Show video
		cv.imshow(window_name, bordered_frame)
		cv.waitKey(10)

	
if __name__ == '__main__':
	main()