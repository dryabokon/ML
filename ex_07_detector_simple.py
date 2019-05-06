# ----------------------------------------------------------------------------------------------------------------------
import sys
sys.path.insert(0, '../tools/')
import time
import cv2
import numpy
import detector_YOLO_simple
import tools_CNN_view
import tools_image
# ----------------------------------------------------------------------------------------------------------------------
def convert_to_Keras_simple():
    D = detector_YOLO_simple.detector_YOLO_simple(None)
    filename_config  = 'data/ex70/darknet_to_keras/yolov3-tiny.cfg'
    filename_weights = 'data/ex70/darknet_to_keras/yolov3-tiny.weights'
    filename_output  = 'data/ex70/darknet_to_keras/yolov3a-tiny.h5'
    D.darknet_to_keras(filename_config, filename_weights, filename_output)
    return
# ----------------------------------------------------------------------------------------------------------------------
def draw_metadata():
    D = detector_YOLO_simple.detector_YOLO_simple('data/ex70/yolov3a-tiny.h5')
    D.metadata_to_image('data/ex70/metadata/','data/output/')
    return
# ----------------------------------------------------------------------------------------------------------------------
def run_YOLO_simple():

    D = detector_YOLO_simple.detector_YOLO_simple('data/ex70/yolov3a.h5')
    D.process_image('data/ex70/Image.jpg', 'data/output/output_simple.jpg')

    return
# ----------------------------------------------------------------------------------------------------------------------
def run_YOLO_on_file():
    D = detector_YOLO_simple.detector_YOLO_simple('data/ex70/yolov3a-tiny.h5')
    D.process_image('data/ex70/bike/Image.png', 'data/output/res.png')
    return
# ----------------------------------------------------------------------------------------------------------------------
def run_YOLO_on_file_debug():

    D = detector_YOLO_simple.detector_YOLO_simple('data/ex70/yolov3a-tiny.h5')
    #D.process_image_debug('data/ex70/racoons/raccoon-1.jpg', 'data/output/')
    D.process_image_debug('data/ex70/bike/image.png', 'data/output/')

    return
# ----------------------------------------------------------------------------------------------------------------------
def run_YOLO_on_folder():

    #D = detector_YOLO_simple.detector_YOLO_simple('data/ex70/yolov3a-tiny.h5')
    D = detector_YOLO_simple.detector_YOLO_simple('data/ex70/racoon_model_2019_5_6.h5')
    start_time = time.time()
    D.process_folder('data/ex70/racoons_all/', 'data/output/')
    print('%s sec\n\n' % (time.time() - start_time))
    return
# ----------------------------------------------------------------------------------------------------------------------
def do_train_simple():
    file_annotations = 'data/ex70/racoons_all/annotation_racoons.txt'
    path_out = 'data/output/'

    D = detector_YOLO_simple.detector_YOLO_simple('data/ex70/yolov3a-tiny.h5')
    D.do_train_tiny_last_layer_only2(file_annotations, path_out)
    return
# ----------------------------------------------------------------------------------------------------------------------
def draw_annotation_boxes():

    file_annotations = 'data/ex70/bike/markup.txt'
    path_out = 'data/output/'

    D = detector_YOLO_simple.detector_YOLO_simple('data/ex70/yolov3a-tiny.h5')
    D.draw_annotation_boxes(file_annotations, path_out)

    return
# ----------------------------------------------------------------------------------------------------------------------
def draw_YOLO_model():

    D = detector_YOLO_simple.detector_YOLO_simple('data/ex70/racoon_model_2019_5_5.h5')
    tools_CNN_view.visualize_layers(D.model, 'data/ex70/bike/Image.png', 'data/output/')
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    run_YOLO_on_file()
    #run_YOLO_on_file_debug()
    #do_train_simple()
    #run_YOLO_on_folder()
    #run_YOLO_on_file_debug()
    #draw_YOLO_model()








