import os
import face_recognition
import cv2
import numpy as np
from PIL import Image, ImageDraw

from gfd.py.video.capture import VideoCaptureThreading

faces_jpg_path='data/faces_jpg'
people_list = os.listdir(faces_jpg_path)
name_list=[]
encoding_list=[]
for i in people_list:
    #print(i)
    faces_per_person=os.listdir(faces_jpg_path+'/'+i)
    #print(faces_per_person)
    for j in faces_per_person:
        #print(faces_jpg_path+'/'+i+'/'+j)
        image=face_recognition.load_image_file(faces_jpg_path+'/'+i+'/'+j)
        name_list.append(i)
        encoding_list.append(face_recognition.face_encodings(image)[0])
#print(name_list)
#print(len(encoding_list))

#video_capture=cv2.VideoCapture('rtsp://admin:gaozhu123@192.168.31.188:554/stream1')
video_capture = VideoCaptureThreading('rtsp://admin:********')
video_capture.start()
video_capture.set(5,15)
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces_jpg and face encodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame,model='cnn')
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    face_landmarks_list = face_recognition.face_landmarks(rgb_frame,face_locations,'large')
    for face_landmarks in face_landmarks_list:

        # Print the location of each facial feature in this image
        # for facial_feature in face_landmarks.keys():
        #     print("The {} in this face has the following points: {}".format(facial_feature,
        #                                                                     face_landmarks[facial_feature]))

        # Let's trace out each facial feature in the image with a line!
        for facial_feature in face_landmarks.keys():
            for point in face_landmarks[facial_feature]:
                cv2.circle(img=frame,center=point,radius=1,color=(0,0,255),thickness=4)

    # Show the picture

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(encoding_list, face_encoding,tolerance=0.55)

        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(encoding_list, face_encoding)
        #print(face_distances)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = name_list[best_match_index]
            #print(name)
        # Draw a box around the face
        #cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        #cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    frame=cv2.resize(frame,None,fx=0.75,fy=0.75)
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        video_capture.stop()
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
