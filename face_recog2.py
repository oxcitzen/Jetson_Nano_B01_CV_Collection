import face_recognition
import cv2


don_face = face_recognition.load_image_file('/home/wei/Desktop/demo_images/known/Donald Trump.jpg')
don_encode = face_recognition.face_encodings(don_face)[0]

nancy_face = face_recognition.load_image_file('/home/wei/Desktop/demo_images/known/Nancy Pelosi.jpg')
nancy_encode = face_recognition.face_encodings(nancy_face)[0]

encodings = [don_encode, nancy_encode]

names = ['Trump', 'Nancy Pelosi']

font = cv2.FONT_HERSHEY_SIMPLEX 
test_image = face_recognition.load_image_file('/home/wei/Desktop/demo_images/unknown/u11.jpg')

face_position = face_recognition.face_locations(test_image)
all_encodings = face_recognition.face_encodings(test_image, face_position)

#print(all_encodings)

test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)

for (row1, col1, row2, col2), face_encoding in zip(face_position, all_encodings):
    name = 'Unknown'
    matches = face_recognition.compare_faces(encodings, face_encoding)

    if True in matches:
        first_match_index = matches.index(True)
        name = names[first_match_index]

    cv2.rectangle(test_image, (col2, row2), (col1, row1), (0 , 0, 255), 2)
    cv2.putText(test_image, name, (col2, row1-6), font, 0.75, (0, 255, 255), 1)


cv2.imshow('my_window', test_image)
cv2.moveWindow('my_window', 0, 0)


if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()
