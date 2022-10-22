from mod_image.image import equalizeHist_rgb
from imutils import face_utils
import dlib
import cv2 as cv


def detect_landmarks(img, mod):

	img = equalizeHist_rgb(img)

	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(mod)

	faces = detector(img, 1)
	for (i, face) in enumerate(faces):
		shape = predictor(img, face)
		shape = face_utils.shape_to_np(shape)
		(x, y, w, h) = face_utils.rect_to_bb(face)
		cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
		cv.putText(img, "Face #{}".format(i + 1), (x - 10, y - 10),
			cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		for (x, y) in shape:
			cv.circle(img, (x, y), 1, (0, 0, 255), -1)

	return img