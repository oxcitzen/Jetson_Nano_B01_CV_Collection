import cv2
print(cv2.__version__)
dispW=640
dispH=480
flip=2
#Uncomment These next Two Line for Pi Camera
#camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
#cam= cv2.VideoCapture(camSet)
 
#Or, if you have a WEB cam, uncomment the next line
#(If it does not work, try setting to '1' instead of '0')

face_cascade = cv2.CascadeClassifier("./cascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("./cascades/haarcascade_eye.xml")

cam=cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey, 1.3, 5)
    
    
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        #Region of interest
        roi_grey = grey[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Detect eyes in the face region
        eyes = eye_cascade.detectMultiScale(roi_grey)
        
        for (x_i, y_i, w_i, h_i) in eyes:
            cv2.rectangle(roi_color, (x_i, y_i), (x_i+w_i, y_i+h_i), (0, 255, 0), 2)
            
        
    
    cv2.imshow('nanoCam',frame)
    cv2.moveWindow('nanoCam', 0, 0)
    if cv2.waitKey(1)==ord('q'):
        break
    
cam.release()
cv2.destroyAllWindows()