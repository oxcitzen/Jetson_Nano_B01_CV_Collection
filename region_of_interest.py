import cv2

go_flag = 0

def mouse_click(event, x, y, flags, params):
    global x1, y1, x2, y2
    global go_flag
    
    if event == cv2.EVENT_LBUTTONDOWN:
        x1 = x
        y1 = y
        go_flag = 0
            
    if event == cv2.EVENT_LBUTTONUP:
        x2 = x
        y2 = y
        go_flag = 1
         
        

cv2.namedWindow('nanoCam')
cv2.setMouseCallback('nanoCam', mouse_click)
dispW=640
dispH=480
flip=2

#Un omment These next Two Line for Pi Camera

camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method=0 ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam= cv2.VideoCapture(camSet)


#cam=cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    cv2.imshow('nanoCam',frame)
    
    if go_flag == 1:
        frame == cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
        
        if y1<y2 and x1<x2:
            roi = frame[y1:y2,x1:x2]  #right bottom
        if y1<y2 and x2<x1:
            roi = frame[y1:y2,x2:x1]  #left bottom
        if y2<y1 and x1<x2:
            roi = frame[y2:y1,x1:x2]   #right up    
        if y2<y1 and x2<x1:
            roi = frame[y2:y1,x2:x1]   #leftqqq up
            
        cv2.imshow('Copy_ROI', roi)
        
    cv2.moveWindow('nanoCam', 0, 0)
    
    if cv2.waitKey(1)==ord('q'):
        break
    
cam.release()
cv2.destroyAllWindows()