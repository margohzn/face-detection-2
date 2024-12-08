import cv2

haarcascade = "haarcascade_frontalface_default.xml"
face_haarcascade = cv2.CascadeClassifier(haarcascade)

web_cam = cv2.VideoCapture(0)

while True:

    status, frame = web_cam.read()
    greyscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_frame = face_haarcascade.detectMultiScale(greyscale_image)
    for (x,y,w,h) in faces_frame:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (150,255,180), 7)

    cv2.imshow("Face detection in video", frame)
    if cv2.waitKey() == 27:
        break


cv2.destroyAllWindows()