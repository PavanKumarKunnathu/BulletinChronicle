import cv2
import face_recognition
import sys
def take_picture():
    print("scanning face")
    cap=cv2.VideoCapture(0)
    ret,frame=cap.read()
    cv2.imwrite('picture.jpg',frame)
    cv2.destroyAllWindows()
    cap.release()
    print("face scan complete")
def analyse_user():
    print("Anlyxing face")
    # C:\Users\HP\PycharmProjects\Bulletin\mysite\users\media\data\deeraj.JPG
    prof_img="pavankumar.jpg"
    # mysite / users / media / data / viswanath.jpg
    fp="mysite/users/media/data/"+prof_img
    # baseimg=face_recognition.load_image_file("C:/Users/HP/OneDrive/Pictures/MainProject/users/kiran.jpg")
    baseimg = face_recognition.load_image_file(fp)
    baseimg=cv2.cvtColor(baseimg,cv2.COLOR_BGR2RGB)

    myface=face_recognition.face_locations(baseimg)[0]
    encodemyface=face_recognition.face_encodings(baseimg)[0]

    cv2.rectangle(baseimg,(myface[3],myface[0]),(myface[1],myface[2]),(255,0,255),2)

    cv2.imshow("Test",baseimg)
    cv2.waitKey(0)

    sampleimg = face_recognition.load_image_file("picture.jpg")
    sampleimg = cv2.cvtColor(sampleimg, cv2.COLOR_BGR2RGB)

    samplefacetest = face_recognition.face_locations(sampleimg)[0]
    try:
        encodesamplefacetest = face_recognition.face_encodings(sampleimg)[0]
        cv2.rectangle(sampleimg, (samplefacetest[3], samplefacetest[0]), (samplefacetest[1], samplefacetest[2]), (255, 0, 255), 2)
        cv2.imshow("Test", sampleimg)
        cv2.waitKey(0)
    except IndexError as e:
        print("index error . Authrntication Faled")
        sys.exit()
    result=face_recognition.compare_faces([encodemyface],encodesamplefacetest)
    print(result)
    output=str(result)
    if output=="[True]":
        print("user authenticate")
    else:
        print("authentication failed")
take_picture()
analyse_user()


    # cv2.rectangle(baseimg, (myface[3], myface[0]), (myface[1], myface[2]), (255, 0, 255), 2)




