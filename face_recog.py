import face_recognition
import cv2


#use full path with jetson nano
image = face_recognition.load_image_file('/home/wei/Desktop/demo_images/unknown/u3.jpg')

face_locations = face_recognition.face_locations(image) # Coordinates of faces
print(face_locations)

image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

for (row1, col1, row2, col2) in face_locations:
    cv2.rectangle(image_bgr, (col1, row1), (col2, row2), (0, 0, 255), 2)

cv2.imshow('my_window', image_bgr)
cv2.moveWindow('my_window', 0, 0)


if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()


