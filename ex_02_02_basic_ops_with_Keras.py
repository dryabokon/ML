from keras.models import Sequential, Model
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D, ZeroPadding2D, Conv2D, Conv2DTranspose, Conv3DTranspose
from keras.layers import BatchNormalization, Input
from keras.layers.core import Flatten, Dense, Dropout,Activation
from keras import backend as K
import numpy
import cv2
K.set_image_dim_ordering('tf')  #(224,224,3)
# ----------------------------------------------------------------------------------------------------------------------
import tools_CNN_view
# ----------------------------------------------------------------------------------------------------------------------
def do_convulution(image):

    model = Sequential()
    model.add(Conv2D(filters=16,kernel_size=(2,2),strides=(1, 1), padding='valid',input_shape=image.shape))
    model.layers[0].set_weights(tools_CNN_view.construct_filters_2x2())
    output = Model(inputs=model.input, outputs=model.output).predict(numpy.array([image]))[0]
    return output #(N,N,64)
# ----------------------------------------------------------------------------------------------------------------------
def _do_convulution_tf(image):
    K.set_image_dim_ordering('th')
    model = Sequential()
    image = image.transpose((2, 0, 1))  #(3,224,224)
    model.add(Conv2D(filters = 64, kernel_size=(3,3),strides=(2, 2), padding='valid', input_shape=image.shape))
    output = Model(inputs=model.input, outputs=model.output).predict(numpy.array([image]))[0]
    output = output.transpose((1, 2, 0)) #(64,N,N)
    return output
# ----------------------------------------------------------------------------------------------------------------------
def do_maxpool(image):
    model = Sequential()
    #model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='same',input_shape=image.shape,data_format='channels_last'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same',input_shape=image.shape,data_format='channels_last'))
    output = Model(inputs=model.input, outputs=model.output).predict(numpy.array([image]))[0]
    return output
# ----------------------------------------------------------------------------------------------------------------------
def do_avgpool(image):
    model = Sequential()
    model.add(AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid',
                           input_shape=(image.shape[0], image.shape[1], image.shape[2]),data_format='channels_last'))

    output = Model(inputs=model.input, outputs=model.output).predict(numpy.array([image]))[0]
    return output #(N,N,64)
# ----------------------------------------------------------------------------------------------------------------------
def do_flatten(image_2d):

    layer_00 = Input(shape=(image_2d.shape[0], image_2d.shape[1]))
    layer_01 = Flatten()(layer_00)
    output = Model(inputs=layer_00, outputs=layer_01).predict(numpy.array([image_2d]))[0]
    return output
# ----------------------------------------------------------------------------------------------------------------------
def do_dropout(image):
    model = Sequential()
    model.add(ZeroPadding2D(input_shape=(image.shape[0], image.shape[1], image.shape[2])))
    model.add(Dropout(rate=0.5))
    model.add(Conv2D(filters=16,kernel_size=(2,2),strides=(1, 1), padding='valid',input_shape=image.shape))
    model.layers[2].set_weights(tools_CNN_view.construct_filters_2x2(16))

    output = Model(inputs=model.input, outputs=model.output).predict(numpy.array([image]))[0]
    return output
# ----------------------------------------------------------------------------------------------------------------------
def do_dense(image):
    model = Sequential()
    model.add(Dense(units=1,activation='relu',input_shape=image.shape))
    w = model.layers[0].get_weights()
    output = Model(inputs=model.input, outputs=model.output).predict(numpy.array([image]))[0]
    return output.reshape(output.shape[0],output.shape[1])
# ----------------------------------------------------------------------------------------------------------------------
def example_convolution(filename_in,filename_out):
    image_in = tools_CNN_view.normalize(cv2.resize(cv2.imread(filename_in),(224,224)).astype(numpy.float32))
    tensor_out = do_convulution(image_in)
    tensor_out = tools_CNN_view.scale(tensor_out)
    image_out = tools_CNN_view.tensor_gray_3D_to_image(tensor_out,do_colorize=False)

    #image_out = tools_image.hitmap2d_to_viridis(image_out)
    cv2.imwrite(filename_out,image_out)
    return
# ----------------------------------------------------------------------------------------------------------------------
def example_maxpool(filename_in,filename_out):
    image_in = cv2.resize(cv2.imread(filename_in),(224,224)).astype(numpy.float32)
    image_out = do_maxpool(image_in)
    cv2.imwrite(filename_out, image_out)
    return
# ----------------------------------------------------------------------------------------------------------------------
def example_avgpool(filename_in,filename_out):
    image_in = cv2.resize(cv2.imread(filename_in),(224,224)).astype(numpy.float32)
    image_out = do_avgpool(image_in)
    cv2.imwrite(filename_out, image_out)
    return
# ----------------------------------------------------------------------------------------------------------------------
def example_flatten(filename_in,filename_out):
    image_in = cv2.imread(filename_in,0).astype(numpy.float32)
    array_out = do_flatten(image_in)
    #image_out = tools_CNN_view.tensor_gray_1D_to_image(array_out)
    cv2.imwrite(filename_out, array_out)
    return
# ----------------------------------------------------------------------------------------------------------------------
def example_dropout(filename_in,filename_out):
    image_in = cv2.imread(filename_in).astype(numpy.float32)
    tensor_out = do_dropout(image_in)
    image_out = tools_CNN_view.tensor_gray_3D_to_image(tensor_out, do_colorize=False)
    cv2.imwrite(filename_out, image_out)
    return
# ----------------------------------------------------------------------------------------------------------------------
def example_dense(filename_in,filename_out):
    image_in = cv2.imread(filename_in).astype(numpy.float32)
    image_out = do_dense(image_in)
    cv2.imwrite(filename_out, image_out)
    return
# ----------------------------------------------------------------------------------------------------------------------
def sample_by_channel(image0):

    images = []
    depth = image0.shape[2]

    for d in range(0,depth):
        image = image0.copy()
        for c in range(0,depth):
            image[:,:,c]=image0[:,:,d]
        images.append(image)

    images = numpy.array(images)

    #for i in range(0, images.shape[0]):
        #cv2.imwrite('data/output/mid_%d.png' % i, images[i])

    return images
# ----------------------------------------------------------------------------------------------------------------------
def combine_by_channels(images):

    if images.shape[0]==1:
        return images[0]

    depth = images.shape[0]
    result = numpy.zeros(images[0].shape,numpy.float32)

    for i in range(0,depth):
        result[:,:,i]+=numpy.average(images[i],axis=2)


    return result
# ----------------------------------------------------------------------------------------------------------------------
def adjust_channels(images_cand):

    images_adjusted = images_cand.copy()

    if images_cand.shape[0]!=1:
        for i in range(0,images_cand.shape[0]):
            img = numpy.average(images_adjusted[i],axis=2)
            images_adjusted[i, :, :, 0] = img
            images_adjusted[i, :, :, 1] = img
            images_adjusted[i, :, :, 2] = img

    min_org,max_org = -1.0,+1.0
    min = images_adjusted.min()
    max = images_adjusted.max()
    images_adjusted*= (max_org-min_org)/(max-min)

    return images_adjusted
# ----------------------------------------------------------------------------------------------------------------------
def example_deconv(filename_in,folder_out):

    n=16
    image = cv2.resize(cv2.imread(filename_in), (224, 224)).astype(numpy.float32)
    cv2.imwrite(folder_out + 'input.png', image)
    #images = sample_by_channel(image)
    images = numpy.array([image])

    images = tools_CNN_view.normalize(images)

    layer_00 = Input(shape=image.shape)
    layer_02 = Conv2D         (n, (2, 2), strides=(1, 1), padding='same')(layer_00)
    layer_03 = Conv2DTranspose(3, (2, 2), strides=(1, 1), padding='same')(layer_02)

    model1 = Model(inputs=layer_00, outputs=layer_02)
    model2 = Model(inputs=layer_00, outputs=layer_03)

    #model1.layers[1].set_weights(tools_CNN_view.construct_filters_2x2(n))

    tensor_out = model1.predict(numpy.array(images))
    for t in range (0,tensor_out.shape[0]):
        image_out = tools_CNN_view.tensor_gray_3D_to_image(tools_CNN_view.scale(tensor_out[t]), do_colorize=False)
        cv2.imwrite(folder_out + 'result1_%02d.png'%t, image_out)

    model2.layers[2].set_weights(tools_CNN_view.inverce_weight(model1.layers[1].get_weights(),3))

    tensor_out = model2.predict(numpy.array(images))

    for t in range(0, tensor_out.shape[0]):
        cv2.imwrite(folder_out + 'result2_%02d.png'%t, tools_CNN_view.scale(tensor_out[t]))

    tensor_out = adjust_channels(tensor_out)
    tensor_out = tools_CNN_view.scale(tensor_out)

    for t in range(0, tensor_out.shape[0]):
        cv2.imwrite(folder_out + 'result3_%02d.png'%t, tensor_out[t])

    res = combine_by_channels(tensor_out)
    cv2.imwrite(folder_out + 'result4.png',res)

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #example_convolution('data/ex22/image4.png', 'data/output/ex_convolution1.png')
    #example_maxpool    ('data/ex22/image2.png', 'data/output/ex_maxpool.png')
    #example_avgpool    ('data/ex22/image2.png', 'data/output/ex_avgpool.png')
    #example_flatten    ('data/ex22/image3.png', 'data/output/ex_flatten.png')
    #example_dense      ('data/ex22/image4.png', 'data/output/ex_dense.png')

    example_deconv('data/ex22/image6.png', 'data/output/')


