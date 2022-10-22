import cv2 as cv
from mod_detection.recognition import detect_faces
from mod_detection.landmark import detect_landmarks


class Mywebcam:

    def __init__(self, camera_id = 0, quitkey = 'q', grayscale = False):
        self.quitkey = quitkey
        self.grayscale = grayscale
        self.vo = cv.VideoCapture(camera_id)

    def set_quit(self, quitkey):
        self.quitkey = quitkey

    def open_camera(self):
        while True:
            ret, frame = self.vo.read()

            # gray color option
            if self.grayscale:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            cv.imshow('frame', frame)

            if cv.waitKey(1) & 0xFF == ord(self.quitkey):
                break

    def detect_faces(self):
        while True:
            ret, frame = self.vo.read()

            # gray color option
            if self.grayscale:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            cv.imshow('frame', detect_faces(frame))

            if cv.waitKey(1) & 0xFF == ord(self.quitkey):
                break

    def detect_landmarks(self, mod):
        while True:
            ret, frame = self.vo.read()

            # gray color option
            if self.grayscale:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            cv.imshow('frame', detect_landmarks(frame, mod))

            if cv.waitKey(1) & 0xFF == ord(self.quitkey):
                break

    def clean(self):
        self.vo.release()
        cv.destroyAllWindows()



