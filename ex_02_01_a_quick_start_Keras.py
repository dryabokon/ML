from keras.applications import MobileNet
from keras.applications.xception import Xception
from keras.applications.mobilenet import preprocess_input
import urllib.request
import cv2
import numpy
import json
from keras import backend as K
K.set_image_dim_ordering('tf')
# ----------------------------------------------------------------------------------------------------------------------
import tools_CNN_view
# ----------------------------------------------------------------------------------------------------------------------
#URL = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'
#data = json.loads(urllib.request.urlopen(URL).read().decode())
#class_names = [data['%d'%i][1] for i in range(0,999)]
class_names = tools_CNN_view.class_names
# ----------------------------------------------------------------------------------------------------------------------
def example_predict():

    #CNN = MobileNet()
    CNN = Xception()

    img = cv2.imread('data/ex09-natural/dog/dog_0001.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (299, 299)).astype(numpy.float32)

    prob = CNN.predict(preprocess_input(numpy.array([img])))
    idx = numpy.argsort(-prob[0])[0]
    print(class_names[idx], prob[0, idx])

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    example_predict()