# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
from math import sqrt
import random


#Get root directory path
dirPath = os.path.dirname(os.path.realpath(__file__))
dirPath = os.path.realpath(dirPath)

def detect_proximity(frame, personNet):

    FOCAL_LENGTH = 615

    #Converting Frame to Blob
    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    #Passing Blob through network to detect and predict
    personNet.setInput(blob)
    detections = personNet.forward()

    #Creating dictionaries to store position and coordinates
    pos = {}
    coordinates = {}

    #Loop over the detections
    for i in np.arange(0, detections.shape[2]):

        #Extracting the confidence of predictions
        confidence = detections[0, 0, i, 2]

        #Filtering out weak predictions
        if confidence > 0.4:

            #Extracting the index of the labels from the detection
            object_id = int(detections[0, 0, i, 1])

            #Identifying only Person as detected object
            if(object_id == 15.0):
                
                #Storing bounding box dimensions
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype('int')

                #Draw the prediction on the frame
                #label = 'Person: {:.1f}%'.format(confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (10,255,0), 2)

                y = startY - 15 if startY - 15 > 15 else startY + 15
                #cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (20,255,0), 1)
                   
                #Adding the bounding box coordinates to dictionary
                coordinates[i] = (startX, startY, endX, endY)

                #Extracting Mid point of bounding box
                midX = abs((startX + endX) / 2)
                midY = abs((startY + endY) / 2)
                
                #Calculating height of bounding box
                ht = abs(endY-startY)

                #Calculating distance from camera
                distance = (FOCAL_LENGTH * 165) / ht
                    
                #Mid-point of bounding boxes in cm
                midX_cm = (midX * distance) / FOCAL_LENGTH
                midY_cm = (midY * distance) / FOCAL_LENGTH
                
                #Appending the mid points of bounding box and distance between detected object and camera 
                pos[i] = (midX_cm, midY_cm, distance)

    #print(pos)
    return (pos, coordinates)

def detect_mask(frame, faceNet, maskNet):
    #Converting Frame to Blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
        (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    #print(detections.shape)

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)


    # return a 2-tuple of the face locations and their corresponding locations
    return (locs, preds)

# load our serialized face detector model from disk
modelDirPath = os.path.join(dirPath,"models")
prototxtPath = os.path.join(modelDirPath, "deploy.prototxt")
weightsPath = os.path.join(modelDirPath, "res10_300x300_ssd_iter_140000.caffemodel")
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model(os.path.join(modelDirPath, "facemask_model.h5"))

caffeModelPath = os.path.join(modelDirPath, "SSD_MobileNet.caffemodel")
protoPath = os.path.join(modelDirPath, "SSD_MobileNet_prototxt.txt")

#Loading Caffe Model
print('[Status] Loading Model...')
personNet= cv2.dnn.readNetFromCaffe(protoPath, caffeModelPath)

# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start() 

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    (locs, preds) = detect_mask(frame, faceNet, maskNet)

    # loop over the detected face locations and their corresponding
    # locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # determine the class label and color we'll use to draw
        # the bounding box and text
        label = "Mask Detected" if mask > withoutMask else "No Mask Detected"
        color = (0, 255, 0) if label == "Mask Detected" else (0, 0, 255)

        # include the probability in the label
        #label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        label = "{}".format(label)

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    #Return position and coordinates of object    
    (pos, coordinates) = detect_proximity(frame, personNet)

    #Creating list to store objects with lower threshold distance than required
    proximity = set()

    #Looping over positions of bounding boxes in frame
    for i in pos.keys():
        for j in pos.keys():
            if i < j:
                #Calculating distance between both detected objects
                dist = sqrt(pow(pos[i][0] - pos[j][0], 2) + pow(pos[i][1] - pos[j][1], 2) + pow(pos[i][2] - pos[j][2], 2))

                #Checking threshold distance - 175 cm and adding warning label
                if dist < 175:
                    proximity.add(i)
                    proximity.add(j)
                    cv2.putText(frame, "Maintain Social Distancing", (50,50), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255,0,0), 1, cv2.LINE_AA)
           
    
    for i in pos.keys():
        if i in proximity:
            color = [0,0,255]
        else:
            color = [0,255,0]

        #Drawing rectangle for detected objects
        (x, y, w, h) = coordinates[i]
        cv2.rectangle(frame, (x, y), (w, h), color, 2)

        label = 'Person: {}'.format(i+1)
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (20,255,0), 1)

    # show the output frame
    if not pos:
        cv2.putText(frame, "No Person", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
