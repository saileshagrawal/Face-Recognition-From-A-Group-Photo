import os
import cv2
import pickle
import face_recognition
import numpy as np
import cvzone
import dlib

# Importing student images
studentImgFolderPath = 'Images'
studentImgNameList = os.listdir(studentImgFolderPath)
studentImgList = []
studentIDList = []
for imgName in studentImgNameList:
    studentImgList.append(cv2.imread(os.path.join(studentImgFolderPath,
                                                  imgName)))
    studentIDList.append(os.path.splitext(imgName)[0])
    fileName = f'{studentImgFolderPath}/{imgName}'


def findEncodings(imgList):
    encodingList = []
    for img in imgList:
        # Converting to RGB format as face_recognition module works only
        # with RGB format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoding = face_recognition.face_encodings(img)[0]
        encodingList.append(encoding)
    # end for

    return (encodingList)
# end function findEncodings()

print("Encoding Started (for known faces)...")
known_EncodingList = findEncodings(studentImgList)  # function call
print("Encoding Successful!")
# print(known_EncodingList)

known_EncodingWithIDsList = [known_EncodingList, studentIDList]

# Load the image
image_path = 'testimage/testphoto.jpeg'
image = cv2.imread(image_path)
#print(image)

# Load pre-trained face detector
detector = dlib.get_frontal_face_detector()

# Detect faces in the image
faces = detector(image, 1)

# Loop through each face found in the image
for i, face in enumerate(faces):
    # Extract the coordinates of the face
    x, y, w, h = (face.left(), face.top(), face.width(), face.height())

    # Draw a rectangle around the face
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Crop and resize the face
    cropped_face = image[y:y+h, x:x+w]
    resized_face = cv2.resize(cropped_face, (216, 216))

    img=resized_face
    
    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # down-scaling image
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    # Set a tolerance threshold
    tolerance = 0.49  # Typical values are between 0.5 and 0.8, lower is more strict

# Use face_recognition module to locate the faces
    webCamFaceLoc = face_recognition.face_locations(imgSmall)
    webCamFaceEncoding = face_recognition.face_encodings(imgSmall,
                                                       webCamFaceLoc)
    studentID = []
   # Compare the test photo face-encodings with the saved encodings
    for faceEncode in webCamFaceEncoding:
        faceMatches = face_recognition.compare_faces(known_EncodingList,
                                                     faceEncode,tolerance)
        faceDists = face_recognition.face_distance(known_EncodingList,
                                                   faceEncode)

        # Get the index of the least distant face from the the database
        matchIndex = np.argmin(faceDists)

        if faceMatches[matchIndex]:
            name = studentIDList[matchIndex]  # save id of matched face
            studentID.append(name)
            # cv2.imshow("final photo",img)
            # cv2.waitKey(3000)
            

    for name in studentID:
        print(name,"found")
        # Draw a box around the face
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(image, (x, y +65),
                      (x+w, y+h), (0, 0, 255),cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, name, (x, y+h),
                    font, 0.8, (255, 255, 255), 1)

    # # Display the resulting image
    cv2.imshow('Video', image)
    cv2.waitKey(500)


# # # end main.py
