import cv2
import jetson_inference
import jetson_utils
import sys
import numpy as np


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


cv2.namedWindow('opencv_cam')
cv2.setMouseCallback('opencv_cam', mouse_click)
dispW=640
dispH=480

camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method=0 ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam= cv2.VideoCapture(camSet)


# For jetson inference
cuda_w=1280
cuda_h=720
net = jetson_inference.imageNet('googlenet')
cuda_disp = jetson_utils.videoOutput("display://0")
font = jetson_utils.cudaFont(size=24)


while True:
    ret, frame = cam.read()
    #cv2.imshow('opencv_cam',frame) # show cv2

    # Overlay instructions at the bottom left corner
    instruction_text = "Press 'Q' to quit | Click and drag the mouse to create ROI"
    cv2.putText(frame, instruction_text, (10, dispH - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('opencv_cam',frame) # show cv2

    #Extract ROI
    if go_flag == 1:
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
        
        if y1<y2 and x1<x2:
            roi = frame[y1:y2,x1:x2]  #right bottom
        if y1<y2 and x2<x1:
            roi = frame[y1:y2,x2:x1]  #left bottom
        if y2<y1 and x1<x2:
            roi = frame[y2:y1,x1:x2]   #right up    
        if y2<y1 and x2<x1:
            roi = frame[y2:y1,x2:x1]   #left up


        # cv2 color format (BGR) --> cuda color format (RGBA)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGBA)
        # Convert numpy to CUDA Frame
        cuda_frame = jetson_utils.cudaFromNumpy(roi)

        roi_width, roi_height = roi.shape[1], roi.shape[0]

        # Prediction
        class_id, confidence = net.Classify(cuda_frame, roi_width, roi_height)
        item = net.GetClassDesc(class_id)
        print(f"Object: {item}, Confidence: {confidence:.2f}")

        font.OverlayText(cuda_frame, roi_width, roi_height, item, 0, 0, font.White, font.Blue)
        cuda_disp.Render(cuda_frame)

    #cv2.moveWindow('opencv_cam', 0, 0)
    

    if cv2.waitKey(1)==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()



