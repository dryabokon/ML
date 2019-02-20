from keras.applications import MobileNet
from keras.applications.xception import Xception
from keras.applications.mobilenet import preprocess_input
from keras.models import Model
import cv2
import numpy
from keras import backend as K
K.set_image_dim_ordering('tf')
# ----------------------------------------------------------------------------------------------------------------------
def example_feature_extract_Keras():

    #CNN = Xception()
    CNN = MobileNet()

    img = cv2.imread('data/ex09-natural/dog/dog_0000.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224)).astype(numpy.float32)
    model = Model(inputs=CNN.input, outputs=CNN.output)
    prob  = model.predict(preprocess_input(numpy.array([img])))


    #model = Model(inputs=CNN.input, outputs=CNN.get_layer('avg_pool').output)
    model = Model(inputs=CNN.input, outputs=CNN.get_layer('global_average_pooling2d_1').output)

    feature = model.predict(preprocess_input(numpy.array([img])))

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    example_feature_extract_Keras()