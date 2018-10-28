import cv2
import numpy as np

cap = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip= 0
faceData = []

datasetPath = "./Desktop/"
fileName = input("Enter name of Person:")

while True:
    ret,frame = cap.read()

    if ret == False:
        continue

    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faces = faceCascade.detectMultiScale(frame, 1.3, 5)
    faces = sorted(faces, key=lambda f:f[2]*f[3] )
##    print(faces)

    for face in faces[-1:] :
        x,y,w,h = face

        cv2.rectangle(grayFrame, (x,y), (x+w, y+h), (0,255,200),2)


        offset = 10

        faceCapture = grayFrame[y-offset: y+h + offset, x-offset:x+w+offset]

        faceCapture = cv2.resize(faceCapture, (100,100))

        skip+=1

        if skip%10 == 0:
            faceData.append(faceCapture)
            print(len(faceData))
                  

    cv2.imshow("Frame", grayFrame)
    cv2.imshow("Face Section", faceCapture)


    keyPressed = cv2.waitKey(1) & 0xFF

    if keyPressed == ord('q'):
        break
faceData = np.asarray(faceData)
faceData = faceData.reshape((faceData.shape[0],-1))
print(faceData.shape)

np.save(datasetPath+fileName+'.npy', faceData)

print("Data Succesfully saved at "+datasetPath+fileName+'.npy')

cap.release()
cv2.destroyAllWindows()                            
                        

    
                            

        



    
