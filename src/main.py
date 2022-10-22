from mod_webcam import Mywebcam


if __name__ == '__main__':

    # create a new instance and an opencv VideoCapture object
    cam = Mywebcam.Mywebcam(grayscale = False)
    # display video
    #cam.open_camera()
    #cam.detect_faces()
    cam.detect_landmarks("../pretrained_models/shape_predictor_68_face_landmarks.dat")
    # closing
    cam.clean()