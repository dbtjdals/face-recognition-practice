#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 15:49:42 2020

@author: dbtjdals
"""

import numpy as np
import face_recognition as fr
import cv2

#capture using webcame
video_capture = cv2.VideoCapture(0)

#read image
david_image = fr.load_image_file('david.jpeg')
dad_image = fr.load_image_file('dad.jpeg')
mom_image = fr.load_image_file('mom.jpeg')

#converts image to array
david_face_encoding = fr.face_encodings(david_image)[0]
dad_face_encoding = fr.face_encodings(dad_image)[0]
mom_face_encoding = fr.face_encodings(mom_image)[0]

#collection of known faces
known_face_encodings = [david_face_encoding, dad_face_encoding, mom_face_encoding]
#name that will show up in the box
known_face_names = ["David", "dad", "mom"]

while True: 
    #read frame
    ret, frame = video_capture.read()

    #change colors of the frame to rgb colors
    rgb_frame = frame[:, :, ::-1]

    #check location of the face; has top, right, bottom, left locations
    face_locations = fr.face_locations(rgb_frame)
    
    #face encoding to see which faces are in the frame
    face_encodings = fr.face_encodings(rgb_frame, face_locations)

    #iterate over the encoding and location 
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        #check for matches between known face (picture) vs face in webcam
        matches = fr.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        face_distances = fr.face_distance(known_face_encodings, face_encoding)

        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        
        #create a box over the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        #create another box for the name
        cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    #show image
    cv2.imshow('Webcam_facerecognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
