#https://github.com/cysmith/neural-style-tf/blob/master/neural_style.py
import tensorflow as tf
import numpy
import scipy.io  
import cv2
import tools_IO
# ----------------------------------------------------------------------------------------------------------------------
def get_CNN(input_img, path_mat):

    CNN = {}

    vgg_rawnet     = scipy.io.loadmat(path_mat)
    vgg_layers     = vgg_rawnet['layers'][0]

    CNN['input']   = tf.Variable(numpy.zeros(input_img.shape, dtype=numpy.float32))


    CNN['conv1_1'] = conv_layer('conv1_1', CNN['input']  , W=get_weights(vgg_layers, 0))
    CNN['relu1_1'] = relu_layer('relu1_1', CNN['conv1_1'], b=get_bias(vgg_layers, 0))
    CNN['conv1_2'] = conv_layer('conv1_2', CNN['relu1_1'], W=get_weights(vgg_layers, 2))
    CNN['relu1_2'] = relu_layer('relu1_2', CNN['conv1_2'], b=get_bias(vgg_layers, 2))
    CNN['pool1']   = pool_layer('pool1'  , CNN['relu1_2'])
    CNN['conv2_1'] = conv_layer('conv2_1', CNN['pool1']  , W=get_weights(vgg_layers, 5))
    CNN['relu2_1'] = relu_layer('relu2_1', CNN['conv2_1'], b=get_bias(vgg_layers, 5))
    CNN['conv2_2'] = conv_layer('conv2_2', CNN['relu2_1'], W=get_weights(vgg_layers, 7))
    CNN['relu2_2'] = relu_layer('relu2_2', CNN['conv2_2'], b=get_bias(vgg_layers, 7))
    CNN['pool2']   = pool_layer('pool2'  , CNN['relu2_2'])
    CNN['conv3_1'] = conv_layer('conv3_1', CNN['pool2']  , W=get_weights(vgg_layers, 10))
    CNN['relu3_1'] = relu_layer('relu3_1', CNN['conv3_1'], b=get_bias(vgg_layers, 10))
    CNN['conv3_2'] = conv_layer('conv3_2', CNN['relu3_1'], W=get_weights(vgg_layers, 12))
    CNN['relu3_2'] = relu_layer('relu3_2', CNN['conv3_2'], b=get_bias(vgg_layers, 12))
    CNN['conv3_3'] = conv_layer('conv3_3', CNN['relu3_2'], W=get_weights(vgg_layers, 14))
    CNN['relu3_3'] = relu_layer('relu3_3', CNN['conv3_3'], b=get_bias(vgg_layers, 14))
    CNN['conv3_4'] = conv_layer('conv3_4', CNN['relu3_3'], W=get_weights(vgg_layers, 16))
    CNN['relu3_4'] = relu_layer('relu3_4', CNN['conv3_4'], b=get_bias(vgg_layers, 16))
    CNN['pool3']   = pool_layer('pool3'  , CNN['relu3_4'])
    CNN['conv4_1'] = conv_layer('conv4_1', CNN['pool3']  , W=get_weights(vgg_layers, 19))
    CNN['relu4_1'] = relu_layer('relu4_1', CNN['conv4_1'], b=get_bias(vgg_layers, 19))
    CNN['conv4_2'] = conv_layer('conv4_2', CNN['relu4_1'], W=get_weights(vgg_layers, 21))
    CNN['relu4_2'] = relu_layer('relu4_2', CNN['conv4_2'], b=get_bias(vgg_layers, 21))
    CNN['conv4_3'] = conv_layer('conv4_3', CNN['relu4_2'], W=get_weights(vgg_layers, 23))
    CNN['relu4_3'] = relu_layer('relu4_3', CNN['conv4_3'], b=get_bias(vgg_layers, 23))
    CNN['conv4_4'] = conv_layer('conv4_4', CNN['relu4_3'], W=get_weights(vgg_layers, 25))
    CNN['relu4_4'] = relu_layer('relu4_4', CNN['conv4_4'], b=get_bias(vgg_layers, 25))
    CNN['pool4']   = pool_layer('pool4'  , CNN['relu4_4'])
    CNN['conv5_1'] = conv_layer('conv5_1', CNN['pool4']  , W=get_weights(vgg_layers, 28))
    CNN['relu5_1'] = relu_layer('relu5_1', CNN['conv5_1'], b=get_bias(vgg_layers, 28))
    CNN['conv5_2'] = conv_layer('conv5_2', CNN['relu5_1'], W=get_weights(vgg_layers, 30))
    CNN['relu5_2'] = relu_layer('relu5_2', CNN['conv5_2'], b=get_bias(vgg_layers, 30))
    CNN['conv5_3'] = conv_layer('conv5_3', CNN['relu5_2'], W=get_weights(vgg_layers, 32))
    CNN['relu5_3'] = relu_layer('relu5_3', CNN['conv5_3'], b=get_bias(vgg_layers, 32))
    CNN['conv5_4'] = conv_layer('conv5_4', CNN['relu5_3'], W=get_weights(vgg_layers, 34))
    CNN['relu5_4'] = relu_layer('relu5_4', CNN['conv5_4'], b=get_bias(vgg_layers, 34))
    CNN['pool5']   = pool_layer('pool5'  , CNN['relu5_4'])
    return CNN
# ----------------------------------------------------------------------------------------------------------------------
def conv_layer(layer_name, layer_input, W):
    return tf.nn.conv2d(layer_input, W, strides=[1, 1, 1, 1], padding='SAME')
# ----------------------------------------------------------------------------------------------------------------------
def relu_layer(layer_name, layer_input, b):
    return tf.nn.relu(layer_input + b)
# ----------------------------------------------------------------------------------------------------------------------
def pool_layer(layer_name, layer_input):

    pool = tf.nn.avg_pool(layer_input, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
    #pool = tf.nn.max_pool(layer_input, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
    return pool
# ----------------------------------------------------------------------------------------------------------------------
def get_weights(vgg_layers, i):
    weights = vgg_layers[i][0][0][2][0][0]
    W = tf.constant(weights)
    return W
# ----------------------------------------------------------------------------------------------------------------------
def get_bias(vgg_layers, i):
    bias = vgg_layers[i][0][0][2][0][1]
    b = tf.constant(numpy.reshape(bias, (bias.size)))
    return b
# ----------------------------------------------------------------------------------------------------------------------
def content_layer_loss(p, x):
    content_loss_function = 1
    _, h, w, d = p.get_shape()
    M = h.value * w.value
    N = d.value
    if content_loss_function   == 1:
        K = 1. / (2. * N**0.5 * M**0.5)
    elif content_loss_function == 2:
        K = 1. / (N * M)
    elif content_loss_function == 3:
        K = 1. / 2.
    loss = K * tf.reduce_sum(tf.pow((x - p), 2))
    return loss
# ----------------------------------------------------------------------------------------------------------------------
def style_layer_loss(a, x):
    _, h, w, d = a.get_shape()
    M = h.value * w.value
    N = d.value
    A = gram_matrix(a, M, N)
    G = gram_matrix(x, M, N)
    loss = (1./(4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((G - A), 2))
    return loss
# ----------------------------------------------------------------------------------------------------------------------
def gram_matrix(x, area, depth):
    F = tf.reshape(x, (area, depth))
    G = tf.matmul(tf.transpose(F), F)
    return G
# ----------------------------------------------------------------------------------------------------------------------
def get_tensor_style_loss(sess, CNN, image):
    style_loss=0.0
    sess.run(CNN['input'].assign(image))
    style_loss += style_layer_loss(tf.convert_to_tensor(sess.run(CNN['relu1_1'])), CNN['relu1_1'])
    style_loss += style_layer_loss(tf.convert_to_tensor(sess.run(CNN['relu2_1'])), CNN['relu2_1'])
    style_loss += style_layer_loss(tf.convert_to_tensor(sess.run(CNN['relu3_1'])), CNN['relu3_1'])
    style_loss += style_layer_loss(tf.convert_to_tensor(sess.run(CNN['relu4_1'])), CNN['relu4_1'])
    style_loss += style_layer_loss(tf.convert_to_tensor(sess.run(CNN['relu5_1'])), CNN['relu5_1'])

    return 0.2*style_loss/5.0
# ----------------------------------------------------------------------------------------------------------------------
def get_tensor_cntnt_loss(sess, CNN, image):
    sess.run(CNN['input'].assign(image))
    tensor = content_layer_loss(tf.convert_to_tensor(sess.run(CNN['conv4_2'])), CNN['conv4_2'])
    return tensor
# ----------------------------------------------------------------------------------------------------------------------
def preprocess(img):
    imgpre = numpy.copy(img)
    imgpre = imgpre[...,::-1]
    imgpre = imgpre[numpy.newaxis, :, :, :]
    imgpre -= numpy.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
    return imgpre.astype(numpy.float32)
# ----------------------------------------------------------------------------------------------------------------------
def postprocess(img):
    imgpost = numpy.copy(img)
    imgpost += numpy.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
    imgpost = imgpost[0]
    imgpost = numpy.clip(imgpost, 0, 255).astype('uint8')
    imgpost = imgpost[...,::-1]
    return imgpost
# ----------------------------------------------------------------------------------------------------------------------
def get_noise_image(noise_ratio, content_img):
    numpy.random.seed(124)
    noise_img = numpy.random.uniform(-20., 20., content_img.shape).astype(numpy.float32)
    img = noise_ratio * noise_img + (1.-noise_ratio) * content_img
    return img
# ----------------------------------------------------------------------------------------------------------------------
def stylize(image_cntnt, image_style,path_mat,path_out):

    tools_IO.remove_files(path_out,create=True)

    image_style = cv2.resize(image_style, dsize=(image_cntnt.shape[1], image_cntnt.shape[0])).astype(numpy.float32)
    image_style = preprocess(image_style)
    image_cntnt = preprocess(image_cntnt.astype(numpy.float32))


    content_weight,style_weight,tv_weight = 5.0, 10000, 0.001
    CNN = get_CNN(image_cntnt, path_mat=path_mat)

    sess = tf.Session()


    T_style_loss = get_tensor_style_loss(sess, CNN, image_style)
    T_cntnt_loss = get_tensor_cntnt_loss(sess, CNN, image_cntnt)
    T_varns_loss = tf.image.total_variation(CNN['input'])

    T_total_loss  = content_weight*T_cntnt_loss + style_weight*T_style_loss + tv_weight*T_varns_loss

    adam = tf.train.AdamOptimizer(1e0)
    optimizer = adam.minimize(T_total_loss)

    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(CNN['input'].assign(image_cntnt))
    iterations = 0
    while (True):
      sess.run(optimizer)
      if (iterations % 10 == 0):
        output_img = sess.run(CNN['input'])
        cv2.imwrite(path_out + 'res_%03d.jpg' % iterations, postprocess(output_img))
        #cv2.imwrite(path_out + 'clr_%03d.jpg' % iterations, postprocess(convert_to_original_colors(image_cntnt.copy(), output_img)))
        print('.')
      iterations += 1

    return
# ----------------------------------------------------------------------------------------------------------------------
def convert_to_original_colors(content_img, stylized_img):
    color_convert_type = 'yuv'
    content_img  = postprocess(content_img)
    stylized_img = postprocess(stylized_img)
    if color_convert_type == 'yuv':
        cvt_type = cv2.COLOR_BGR2YUV
        inv_cvt_type = cv2.COLOR_YUV2BGR
    elif color_convert_type == 'ycrcb':
        cvt_type = cv2.COLOR_BGR2YCR_CB
        inv_cvt_type = cv2.COLOR_YCR_CB2BGR
    elif color_convert_type == 'luv':
        cvt_type = cv2.COLOR_BGR2LUV
        inv_cvt_type = cv2.COLOR_LUV2BGR
    elif color_convert_type == 'lab':
        cvt_type = cv2.COLOR_BGR2LAB
        inv_cvt_type = cv2.COLOR_LAB2BGR
    content_cvt = cv2.cvtColor(content_img, cvt_type)
    stylized_cvt = cv2.cvtColor(stylized_img, cvt_type)
    c1, _, _ = cv2.split(stylized_cvt)
    _, c2, c3 = cv2.split(content_cvt)
    merged = cv2.merge((c1, c2, c3))
    dst = cv2.cvtColor(merged, inv_cvt_type).astype(numpy.float32)
    dst = preprocess(dst)
    return dst
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    path = '../_images/ex32b/'
    path_mat = path + 'vgg/imagenet-vgg-verydeep-19e.mat'
    image_cntnt = cv2.imread(path + '/in/kiev.jpg')
    image_style = cv2.imread(path + '/style/vGogh.jpg')
    path_out = path + 'out_haddock/'

    stylize(image_cntnt, image_style,path_mat,path_out)

