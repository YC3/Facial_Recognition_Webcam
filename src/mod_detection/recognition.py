from mod_image.image import equalizeHist_rgb
import cv2 as cv


def detect_faces(img):

    img = equalizeHist_rgb(img)

    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(img)

    for (x, y, w, h) in faces:

        center = (x + w//2, y + h//2)
        img = cv.ellipse(img, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        faceROI = img[y:y + h, x:x + w]

        if 1:
            eyes_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
            eyes = eyes_cascade.detectMultiScale(faceROI)
            for (x2, y2, w2, h2) in eyes:
                eye_center = (x + x2 + w2//2, y + y2 + h2//2)
                radius = int(round((w2 + h2)*0.25))
                img = cv.circle(img, eye_center, radius, (255, 0, 0 ), 4)

    return img




