import face_recognition
import cv2
import os
import pickle
import time


'''

The scale_factor variable is used to resize the video frames before they are processed for face detection. 
In this example, scale_factor = 0.25 means each frame is reduced to 25% of its original width and height.

Smaller images mean fewer pixels to process, which makes face detection and encoding faster. 
This is especially helpful for real-time applications where high frame rates are desirable.

After processing, the detected face coordinates (in the resized frame) 
are scaled back to the original size by multiplying each coordinate by 1 / scale_factor 

This ensures that the bounding boxes and labels are drawn correctly on the original frame.
'''

fps_report = 0
scale_factor = .25


Encodings=[]
Names=[]

with open('train.pkl','rb') as f:
    Names=pickle.load(f)
    Encodings=pickle.load(f)
font=cv2.FONT_HERSHEY_SIMPLEX


dispW=640
dispH=480
flip=2

# For pi camera with jetson nano
camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method=0 ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam= cv2.VideoCapture(camSet)

#cam= cv2.VideoCapture(0)
time_stamp = time.time()
while True:

    _,frame=cam.read()
    frameSmall=cv2.resize(frame,(0,0),fx=scale_factor,fy=scale_factor)
    frameRGB=cv2.cvtColor(frameSmall,cv2.COLOR_BGR2RGB)

    # faster computational with cnn
    facePositions=face_recognition.face_locations(frameRGB,model='cnn')
    
    
    allEncodings=face_recognition.face_encodings(frameRGB,facePositions)

    for (top,right,bottom,left),face_encoding in zip(facePositions,allEncodings):

        name='Unkown Person'
        matches=face_recognition.compare_faces(Encodings,face_encoding)

        if True in matches:
            first_match_index=matches.index(True)
            name=Names[first_match_index]

        top= int(top / scale_factor)
        right= int(right / scale_factor)
        bottom= int(bottom / scale_factor)
        left= int(left / scale_factor)

        #Box the detected face
        cv2.rectangle(frame,(left,top),(right, bottom),(0,0,255),2)
        cv2.putText(frame, name,(left,top-6),font,.75,(0,0,255),2)

    dt = time.time() - time_stamp
    fps = 1 / dt
    fps_report = 0.90 * fps_report + 0.1 * fps
    time_stamp = time.time()

    # Display Frame Per Second
    cv2.rectangle(frame, (0, 0), (100, 40), (0, 0, 255), -1)
    cv2.putText(frame, str(round(fps_report, 1)) + 'fps' , (0, 25), font, 0.75, (0, 255, 255, 2))


    # Show original frame
    cv2.imshow('Picture',frame)
    cv2.moveWindow('Picture',0,0)
    if cv2.waitKey(1)==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()