import cv2

dispW=320
dispH=240
flip=2

def nothing(x):
    pass

# Create the 'blended' window and a trackbar for blending
cv2.namedWindow('blended')
cv2.createTrackbar('blend_v', 'blended', 50, 100, nothing)

cv_logo = cv2.imread('mask.jpg') 
cv_logo = cv2.resize(cv_logo, (320, 240))

# To do thresholing with gray scale 
cv_logo_gray = cv2.cvtColor(cv_logo, cv2.COLOR_BGR2GRAY) 
cv2.imshow('cv_logo_gray', cv_logo_gray)
cv2.moveWindow('cv_logo_gray', 0, 720)

# 255 white, 0 black
# 220 is the thresh value. Pixel below is set to 0 and pixel above is set to max value
# 255 is the max value
_, bg_mask = cv2.threshold(cv_logo_gray, 220, 255, cv2.THRESH_BINARY)
cv2.imshow('BG_mask', bg_mask)
cv2.moveWindow('BG_mask', 330, 440)

# The inversion of bg_mask 
fg_mask = cv2.bitwise_not(bg_mask)
cv2.imshow('FG_mask', fg_mask)
cv2.moveWindow('FG_mask', 330, 720)


# mask 
# Pixels in the mask that are non-zero will have the operation applied to the corresponding pixels in the source image. 
# Pixels in the mask that are zero (black, i.e., 0) will not be affected.  
fg = cv2.bitwise_and(cv_logo , cv_logo, mask=fg_mask)
cv2.imshow('FG', fg)
cv2.moveWindow('FG', 660, 720)



#Uncomment These next Two Line for Pi Camera
#camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
#cam= cv2.VideoCapture(camSet)

cam=cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    
    frame = cv2.resize(frame, (dispW, dispH))
    
    cv2.namedWindow('nanoCam', cv2.WINDOW_NORMAL)
    cv2.imshow('nanoCam',frame)
    cv2.moveWindow('nanoCam', 0, 440)
    
    # mask 
    # Pixels in the mask that are non-zero will have the operation applied to the corresponding pixels in the source image. 
    # Pixels in the mask that are zero (black, i.e., 0) will not be affected.
    
    # Frame with black logo
    bg = cv2.bitwise_and(frame, frame, mask=bg_mask)
    cv2.imshow('BG', bg)
    cv2.moveWindow('BG', 660, 440)
    
    
    # Frame with color logo
    comp_image = cv2.add(bg, fg)
    cv2.imshow('comp_image', comp_image)
    cv2.moveWindow('comp_image', 990, 440)
    
    # get the value from the track bar
    bv1 = cv2.getTrackbarPos('blend_v', 'blended') / 100
    bv2 = 1 - bv1
    
    
    blended = cv2.addWeighted(frame, bv1, cv_logo, bv2, 0)
    cv2.imshow('blended', blended)
    cv2.moveWindow('blended', 990, 720)
    
    # Pixels in the mask that are zero (black, i.e., 0) will not be affected.
    # Black background with pale logo frame
    fore_ground_2 = cv2.bitwise_and(blended , blended, mask=fg_mask)
    cv2.imshow('FG2', fore_ground_2)
    cv2.moveWindow('FG2', 1320, 440)
    

    #comp_final = cv2.addWeighted(frame, 1, fore_ground_2, 2, 2)
    # Frame as background and with addtion of black and pale logo
    comp_final = cv2.add(bg, fore_ground_2)
    cv2.imshow('comp_final', comp_final)
    cv2.moveWindow('comp_final', 1320, 720)
    
    
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
