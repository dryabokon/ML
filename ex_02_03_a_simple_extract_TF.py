import tensorflow_hub as hub
import cv2
import numpy
import tensorflow as tf
# ----------------------------------------------------------------------------------------------------------------------
def example_feature_extract_TF():

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

    feature = module(img).eval(session=sess)

    sess.close()

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    example_feature_extract_TF()