import tensorflow_hub as hub
import urllib.request
import cv2
import numpy
import json
import tensorflow as tf
import CNN_AlexNet_TF
# ----------------------------------------------------------------------------------------------------------------------
URL = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'
data = json.loads(urllib.request.urlopen(URL).read().decode())
class_names = [data['%d'%i][1] for i in range(0,999)]
# ----------------------------------------------------------------------------------------------------------------------
def example_predict_resnet_v2_50():

    module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_50/classification/1")
    img = cv2.imread('data/ex09-natural/dog/dog_0000.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224)).astype(numpy.float32)
    img = numpy.array([img]).astype(numpy.float32) / 255.0
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    outputs = module(dict(images=img), signature="image_classification", as_dict=True)
    prob = outputs["default"].eval(session=sess)[0]
    idx = numpy.argsort(-prob)[0]

    print(class_names[idx], prob[idx])
    sess.close()

    return
# ----------------------------------------------------------------------------------------------------------------------
def example_predict_alexnet():
    CNN = CNN_AlexNet_TF.CNN_AlexNet_TF()
    image = cv2.imread('data/ex09-natural/dog/dog_0000.jpg')
    print(CNN.predict(image))

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    example_predict_resnet_v2_50()
    #example_predict_alexnet()