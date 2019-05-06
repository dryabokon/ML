# ----------------------------------------------------------------------------------------------------------------------
import os
import cv2
# ----------------------------------------------------------------------------------------------------------------------
from keras.applications.xception import Xception
from keras.applications import MobileNet
#from keras import backend as K
#K.set_image_dim_ordering('tf')
# ----------------------------------------------------------------------------------------------------------------------
import classifier_FC_Keras
import CNN_VGG16_Keras
import CNN_AlexNet_TF
import tools_IO
import tools_CNN_view
import detector_YOLO3
import detector_YOLO_simple
# ----------------------------------------------------------------------------------------------------------------------
def visualize_layers_TF_Alexnet():

    filename_input = 'data/ex09-natural/dog/dog_0000.jpg'
    #filename_input = 'data/ex-natural/fruit/fruit_0082.jpg'
    path_output = 'data/output/'

    if not os.path.exists(path_output):
        os.makedirs(path_output)
    else:tools_IO.remove_files(path_output)

    CNN = CNN_AlexNet_TF.CNN_AlexNet_TF()
    image = cv2.imread(filename_input)
    #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    print(CNN.predict(image))

    CNN.visualize_filters(path_output)
    CNN.visualize_layers(filename_input, path_output)

    return
# ----------------------------------------------------------------------------------------------------------------------
def visualize_layers_detector_YOLO3():

    filename_image = 'data/ex70/image.jpg'
    path_out = 'data/output/'

    if not os.path.exists(path_out):
        os.makedirs(path_out)
    else:tools_IO.remove_files(path_out)

    CNN = detector_YOLO3.detector_YOLO3('data/ex70/yolov3-tiny.h5')

    tools_CNN_view.visualize_layers(CNN.model,filename_image, path_out)


    return
# ----------------------------------------------------------------------------------------------------------------------
def visualize_layers_detector_YOLO_simple():

    filename_image = 'data/ex70/image.jpg'
    path_out = 'data/output/'

    if not os.path.exists(path_out):
        os.makedirs(path_out)
    else:tools_IO.remove_files(path_out)

    CNN = detector_YOLO_simple.detector_YOLO_simple('data/ex70/yolov3a-tiny.h5')

    tools_CNN_view.visualize_layers(CNN.model,filename_image, path_out)

    return
# ----------------------------------------------------------------------------------------------------------------------
def visualize_layers_keras_VGG16():

    filename_image = 'data/ex-natural/dog/dog_0000.jpg'
    path_out = 'data/output/'

    if not os.path.exists(path_out):
        os.makedirs(path_out)
    else:tools_IO.remove_files(path_out)

    CNN = CNN_VGG16_Keras.CNN_VGG16_Keras()
    tools_CNN_view.visualize_filters(CNN.model, path_out)
    tools_CNN_view.visualize_layers(CNN.model,filename_image, path_out)

    return
# ----------------------------------------------------------------------------------------------------------------------
def visualize_layers_keras_Xception():

    filename_image = 'data/ex-natural/dog/dog_0000.jpg'
    path_out = 'data/output/'

    if not os.path.exists(path_out):
        os.makedirs(path_out)
    else:tools_IO.remove_files(path_out)

    CNN = Xception()
    tools_CNN_view.visualize_filters(CNN, path_out)
    tools_CNN_view.visualize_layers(CNN,filename_image, path_out)

    return
# ----------------------------------------------------------------------------------------------------------------------
def visualize_layers_keras_MobileNet():
    filename_image = 'data/ex-natural/dog/dog_0000.jpg'
    path_out = 'data/output/'

    if not os.path.exists(path_out):
        os.makedirs(path_out)
    else:
        tools_IO.remove_files(path_out)

    CNN = MobileNet()
    tools_CNN_view.visualize_filters(CNN, path_out)
    tools_CNN_view.visualize_layers(CNN, filename_image, path_out)
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #visualize_layers_TF_Alexnet()
    #visualize_layers_keras_MobileNet()
    #visualize_layers_keras_Xception()
    visualize_layers_detector_YOLO3()
    #visualize_layers_detector_YOLO_simple()