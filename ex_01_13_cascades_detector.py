import cv2
import numpy
import tools_image
# ---------------------------------------------------------------------------------------------------------------------
filename_in  = 'data/ex13/input.jpg'
filename_out = 'data/output/out.jpg'
faceCascade = cv2.CascadeClassifier('data/ex13/haarcascade_frontalface_default.xml')
USE_CAMERA = True


# ---------------------------------------------------------------------------------------------------------------------
def demo_cascade():

    if USE_CAMERA:
        cap = cv2.VideoCapture(0)
    else:
        cap = []
        frame = cv2.imread(filename_in)

    while (True):
        if USE_CAMERA:
            ret, frame = cap.read()

        gray_rgb = tools_image.desaturate(frame)
        faces = faceCascade.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(gray_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('frame', gray_rgb)
        key = cv2.waitKey(1)
        if key & 0xFF == 27:
            break

        if (key & 0xFF == 13) or (key & 0xFF == 32):
            cv2.imwrite(filename_out, gray_rgb)

    if USE_CAMERA:
        cap.release()

    cv2.destroyAllWindows()
    if not USE_CAMERA:
        cv2.imwrite(filename_out, gray_rgb)

    return
# ---------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':


    demo_cascade()
