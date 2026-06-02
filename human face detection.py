import cv2

algorithm = "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(algorithm)

video_detect = "newvideo1.mp4"
video = cv2.VideoCapture(video_detect)

while True:
    ret, img = video.read()

    if not ret:
        break

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray_img, 1.3,17)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (150,255,130), 5)

    img = cv2.resize(img, (1600,900))
    cv2.imshow("Human Face Detection", img)

    key = cv2.waitKey(25)
    if key == 27:
        break

video.release()
cv2.destroyAllWindows()
