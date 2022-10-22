import cv2 as cv


def equalizeHist_rgb(img):

    R, G, B = cv.split(img)

    output1_R = cv.equalizeHist(R)
    output1_G = cv.equalizeHist(G)
    output1_B = cv.equalizeHist(B)

    return cv.merge((output1_R, output1_G, output1_B))