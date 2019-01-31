import cv2
import os
import tools_image
# ---------------------------------------------------------------------------------------------------------------------
path = 'data/ex13_single/'
filename_in  = path + 'pos_single/object_to_detect.jpg'
filename_out = 'data/output/detect_result.jpg'
object_detector= cv2.CascadeClassifier(path + 'output_single/cascade.xml')

# ---------------------------------------------------------------------------------------------------------------------
def train_cascade():
    os.chdir(path)
    os.system('1-create_vec_from_single.bat')#os.system('2-verify_vec_single.bat')
    os.system('3-train_from_single.bat')
    return
# ---------------------------------------------------------------------------------------------------------------------
def test_cascade():
    image = tools_image.desaturate(cv2.imread(filename_in))
    objects, rejectLevels, levelWeights = \
        object_detector.detectMultiScale3(image, scaleFactor=1.05, minSize=(20, 20), outputRejectLevels=True)
    for (x, y, w, h) in objects:cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(filename_out,image)

    return
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #train_cascade()
    test_cascade()

