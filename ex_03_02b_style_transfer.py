import functools
import tensorflow as tf
import numpy
import cv2
import scipy.misc
import scipy.io
import tools_IO
import tools_image
# ----------------------------------------------------------------------------------------------------------------------
WEIGHTS_INIT_STDEV = .1
# ----------------------------------------------------------------------------------------------------------------------
def vgg_net(data_path, input_image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    data = scipy.io.loadmat(data_path)
    mean = data['normalization'][0][0][0]
    mean_pixel = numpy.mean(mean, axis=(0, 1))
    weights = data['layers'][0]

    net = {}
    current = input_image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = numpy.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            current = my_conv_layer(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = my_pool_layer(current)
        net[name] = current

    assert len(net) == len(layers)
    return net
# ----------------------------------------------------------------------------------------------------------------------

def my_conv_layer(input, weights, bias):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),padding='SAME')
    return tf.nn.bias_add(conv, bias)
# ----------------------------------------------------------------------------------------------------------------------

def my_pool_layer(input):
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),padding='SAME')
# ----------------------------------------------------------------------------------------------------------------------

def preprocess(image):
    #return image - numpy.array([123.68 , 116.779, 103.939])
    return image - numpy.array([128 , 128, 128])
# ----------------------------------------------------------------------------------------------------------------------

def unprocess(image):
    return image + numpy.array([128, 128, 128])
# ----------------------------------------------------------------------------------------------------------------------
def get_img(src, img_size=False):
   img = scipy.misc.imread(src, mode='RGB') # misc.imresize(, (256, 256, 3))
   if not (len(img.shape) == 3 and img.shape[2] == 3):
       img = numpy.dstack((img,img,img))
   if img_size != False:
       img = scipy.misc.imresize(img, img_size)
   return img
# ----------------------------------------------------------------------------------------------------------------------
def net(image):
    conv1 = _conv_layer(image, 32, 9, 1)
    conv2 = _conv_layer(conv1, 64, 3, 2)
    conv3 = _conv_layer(conv2, 128, 3, 2)
    resid1 = _residual_block(conv3, 3)
    resid2 = _residual_block(resid1, 3)
    resid3 = _residual_block(resid2, 3)
    resid4 = _residual_block(resid3, 3)
    resid5 = _residual_block(resid4, 3)
    conv_t1 = _conv_tranpose_layer(resid5, 64, 3, 2)
    conv_t2 = _conv_tranpose_layer(conv_t1, 32, 3, 2)
    conv_t3 = _conv_layer(conv_t2, 3, 9, 1, relu=False)
    preds = tf.nn.tanh(conv_t3) * 150 + 255./2
    return preds
# ----------------------------------------------------------------------------------------------------------------------
def _conv_layer(net, num_filters, filter_size, strides, relu=True):
    weights_init = _conv_init_vars(net, num_filters, filter_size)
    strides_shape = [1, strides, strides, 1]
    net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME')
    net = _instance_norm(net)
    if relu:
        net = tf.nn.relu(net)

    return net
# ----------------------------------------------------------------------------------------------------------------------
def _conv_tranpose_layer(net, num_filters, filter_size, strides):
    weights_init = _conv_init_vars(net, num_filters, filter_size, transpose=True)

    batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
    new_rows, new_cols = int(rows * strides), int(cols * strides)
    # new_shape = #tf.pack([tf.shape(net)[0], new_rows, new_cols, num_filters])

    new_shape = [batch_size, new_rows, new_cols, num_filters]
    tf_shape = tf.stack(new_shape)
    strides_shape = [1,strides,strides,1]

    net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')
    net = _instance_norm(net)
    return tf.nn.relu(net)
# ----------------------------------------------------------------------------------------------------------------------
def _residual_block(net, filter_size=3):
    tmp = _conv_layer(net, 128, filter_size, 1)
    return net + _conv_layer(tmp, 128, filter_size, 1, relu=False)
# ----------------------------------------------------------------------------------------------------------------------
def _instance_norm(net, train=True):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
    return scale * normalized + shift
# ----------------------------------------------------------------------------------------------------------------------
def _conv_init_vars(net, out_channels, filter_size, transpose=False):
    _, rows, cols, in_channels = [i.value for i in net.get_shape()]
    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=WEIGHTS_INIT_STDEV, seed=1), dtype=tf.float32)
    return weights_init
# ----------------------------------------------------------------------------------------------------------------------
def get_noise_image(noise_ratio, content_img):
  numpy.random.seed(124)
  noise_img = numpy.random.uniform(-20., 20., content_img.shape).astype(numpy.float32)
  img = noise_ratio * noise_img + (1.-noise_ratio) * content_img
  return img
# ----------------------------------------------------------------------------------------------------------------------
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'
# ----------------------------------------------------------------------------------------------------------------------
def optimize(image_content, image_style, res_prefix,  vgg_path, image_prelim=None):

    learning_rate = 10
    content_weight = 7.5
    style_weight = 100
    tv_weight = 200

    style_features = {}

    batch_shape = (1,image_content.shape[0],image_content.shape[1],3)
    style_shape = (1,) + image_style.shape

    with tf.Graph().as_default(), tf.device('/cpu:0'), tf.Session() as sess:
        style_image = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
        style_image = preprocess(style_image)
        cnn_net = vgg_net(vgg_path, style_image)
        style_pre = numpy.array([image_style])
        for layer in STYLE_LAYERS:
            features = cnn_net[layer].eval(feed_dict={style_image:style_pre})
            features = numpy.reshape(features, (-1, features.shape[3]))
            gram = numpy.matmul(features.T, features) / features.size
            style_features[layer] = gram

    with tf.Graph().as_default(), tf.Session() as sess:
        X_content = tf.placeholder(tf.float32, shape=batch_shape, name="X_content")
        X_pre = preprocess(X_content)

        # precompute content features
        content_features = {}
        content_net = vgg_net(vgg_path, X_pre)
        content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]

        if image_prelim is None:
            im = get_noise_image(0.8, preprocess(image_content)).astype(numpy.float32)
            preds = tf.Variable(tf.convert_to_tensor(numpy.expand_dims(im,axis=0)))
        else:
            im = preprocess(image_prelim).astype(numpy.float32)
            preds = tf.Variable(tf.convert_to_tensor(numpy.expand_dims(im, axis=0)))


        cnn_net = vgg_net(vgg_path, preds)

        content_size = _tensor_size(content_features[CONTENT_LAYER])
        assert _tensor_size(content_features[CONTENT_LAYER]) == _tensor_size(cnn_net[CONTENT_LAYER])
        content_loss = content_weight * (2 * tf.nn.l2_loss(cnn_net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / content_size)

        style_losses = []
        for style_layer in STYLE_LAYERS:
            layer = cnn_net[style_layer]
            bs, height, width, filters = map(lambda i:i.value,layer.get_shape())
            size = height * width * filters
            feats = tf.reshape(layer, (bs, height * width, filters))
            feats_T = tf.transpose(feats, perm=[0,2,1])
            grams = tf.matmul(feats_T, feats) / size
            style_gram = style_features[style_layer]
            style_losses.append(2 * tf.nn.l2_loss(grams - style_gram)/style_gram.size)

        style_loss = style_weight * functools.reduce(tf.add, style_losses)

        # total variation denoising
        tv_y_size = _tensor_size(preds[:,1:,:,:])
        tv_x_size = _tensor_size(preds[:,:,1:,:])
        y_tv = tf.nn.l2_loss(preds[:,1:,:,:] - preds[:,:batch_shape[1]-1,:,:])
        x_tv = tf.nn.l2_loss(preds[:,:,1:,:] - preds[:,:,:batch_shape[2]-1,:])
        tv_loss = tv_weight*2*(x_tv/tv_x_size + y_tv/tv_y_size)

        loss = content_loss + style_loss + tv_loss

        # overall loss
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        sess.run(tf.global_variables_initializer())

        X = image_content.astype(numpy.float32)

        iteration=0
        while (True):
            train_step.run(feed_dict={X_content:[X]})
            _style_loss, _content_loss, _tv_loss, _loss, _preds = sess.run([style_loss, content_loss, tv_loss, loss, preds], feed_dict = {X_content:[X]})
            if (iteration % 10 == 0):
                print(iteration)
                cv2.imwrite(res_prefix+ 'res_%03d.jpg' % iteration,unprocess(_preds)[0])
            iteration+=1

# ----------------------------------------------------------------------------------------------------------------------
def _tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)
# ----------------------------------------------------------------------------------------------------------------------
def apply_style():

    path = '../_images/ex32b/'
    filename_in  = path + 'in/kiev.jpg'
    filename_out = path + 'out/res.jpg'
    chpnt_dir    = path + 'ckpt/muse.ckpt'

    X = cv2.imread(filename_in).astype(numpy.float32)
    X = numpy.array([X])
    sess = tf.Session()

    img_placeholder = tf.placeholder(tf.float32, shape=X.shape, name='img_placeholder')
    preds = net(img_placeholder)
    tf.train.Saver().restore(sess, chpnt_dir)
    res = sess.run(preds, feed_dict={img_placeholder: X})
    cv2.imwrite(filename_out, res[0])
    sess.close()
    return res[0]
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    path = '../_images/ex32b/'
    image_content = cv2.imread(path + '/in/kiev.jpg')
    image_style = cv2.imread(path + '/style/oil.jpg')
    image_prelim = cv2.imread(path + 'res_370.jpg')
    res_prefix = path + 'out_b/'
    vgg_path = path + '/vgg/imagenet-vgg-verydeep-19b.mat'

    #tools_IO.remove_files(res_prefix)
    #optimize(image_content, image_style, res_prefix, vgg_path, image_prelim)


    #apply_style()

    img = cv2.imread('d:\image22.png')
    cv2.imwrite('d:\image23.png',tools_image.canvas_extrapolate(img,457,1200))




