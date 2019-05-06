# ----------------------------------------------------------------------------------------------------------------------
import time
import detector_YOLO3
# ----------------------------------------------------------------------------------------------------------------------
def convert_to_Keras():
    D = detector_YOLO3.detector_YOLO3(None)
    filename_config  = 'data/ex70/darknet_to_keras/yolov3.cfg'
    filename_weights = 'data/ex70/darknet_to_keras/yolov3.weights'
    filename_output  = 'data/ex70/darknet_to_keras/_yolov3.h5'
    D.darknet_to_keras(filename_config, filename_weights, filename_output)
    return
# ----------------------------------------------------------------------------------------------------------------------
def example_YOLO3_on_file():

    D = detector_YOLO3.detector_YOLO3('data/ex70/yolov3-tiny.h5')
    D.process_image('data/ex70/bike/Image.png', 'data/output/res.jpg')

    return
# ----------------------------------------------------------------------------------------------------------------------
def example_YOLO3_on_folder():

    #D = detector_YOLO3.detector_YOLO3('data/ex70/yolov3-tiny.h5')
    D = detector_YOLO3.detector_YOLO3('data/ex70/racoon_model.h5')
    start_time = time.time()
    D.process_folder('data/ex70/octavia/', 'data/output/')
    print('%s sec\n\n' % (time.time() - start_time))
    return
# ----------------------------------------------------------------------------------------------------------------------
def do_train():
    file_annotations = 'data/ex70/annotation_racoons.txt'   #D.prepare_annotation_file('data/ex09-natural/','data/annotation.txt')
    path_out = 'data/output/'
    D = detector_YOLO3.detector_YOLO3('data/ex70/yolov3a-tiny.h5')
    D.do_train_tiny_last_layer_only(file_annotations, path_out)
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    #do_train()
    example_YOLO3_on_file()





