import os
import matplotlib.pyplot as plt
from os import listdir
import fnmatch
import numpy
import cv2
# ----------------------------------------------------------------------------------------------------------------------
import generator_FC_Keras
import tools_IO
import tools_ML
import tools_image
import ex05_visualize_features
# ----------------------------------------------------------------------------------------------------------------------
def do_learn(folder_features,folder_images,folder_output,model_output,resize_H=8, resize_W = 8,grayscale=False):

    patterns = numpy.array([each.split('.')[0] for each in fnmatch.filter(listdir(folder_features), '*.txt')])
    features,Y,filenames = tools_ML.tools_ML(None).prepare_arrays_from_feature_files(folder_features,patterns=patterns,feature_mask='.txt')

    flat_images = []
    for each in filenames:
        local_image = cv2.imread(folder_images + each.split('_')[0] + '/' + each)
        local_image = cv2.resize(local_image,(resize_H,resize_W))
        if grayscale == True:
            local_image = tools_image.desaturate_2d(local_image)

        flat_images.append(local_image.flatten())

    flat_images = numpy.array(flat_images).astype(numpy.float)
    flat_images= flat_images/flat_images.max()
    features= features/(features.max())

    G = generator_FC_Keras.generator_FC_Keras(folder_output)
    G.learn(features.astype(numpy.float),flat_images.astype(numpy.float),grayscale)
    G.save_model(model_output)

    fig = plt.figure(figsize=(12, 6))
    fig.subplots_adjust(hspace=0.01)
    tools_IO.plot_learning_rates1(plt.subplot(1, 2, 1), fig,filename_mat=folder_output + G.name + '_learn_rates.txt')
    tools_IO.plot_learning_rates2(plt.subplot(1, 2, 2), fig,filename_mat=folder_output + G.name + '_learn_rates.txt')
    plt.show()

    return
# ----------------------------------------------------------------------------------------------------------------------
def do_generate(folder_features,filename_model,folder_output,resize_H=8, resize_W = 8,grayscale=False):

    tools_IO.remove_files(folder_output)
    tools_IO.remove_folders(folder_output)
    G = generator_FC_Keras.generator_FC_Keras()
    G.load_model(filename_model)
    shape = (resize_H, resize_W) if grayscale else (resize_H, resize_W, 3)

    patterns = numpy.array([each.split('.')[0] for each in fnmatch.filter(listdir(folder_features), '*.txt')])
    for each in patterns: os.makedirs(folder_output + each)

    features,Y,filenames = tools_ML.tools_ML(None).prepare_arrays_from_feature_files(folder_features, patterns=patterns, feature_mask='.txt')

    generated_images = G.generate(features.astype(numpy.float32),shape=shape)
    for i in range(0,generated_images.shape[0]):
        cv2.imwrite(folder_output + patterns[Y[i]] +'/'+ filenames[i],generated_images[i])

    for pattern in patterns:
        idx = numpy.where(patterns[Y]==pattern)
        feature_avg = numpy.average(features[idx],axis=0)
        generated_image = G.generate(numpy.array([feature_avg]), shape=shape)[0]
        cv2.imwrite(folder_output + pattern + '_average_feature.png', generated_image)

    feature_avg = numpy.average(features, axis=0)
    generated_image = G.generate(numpy.array([feature_avg]), shape=shape)[0]
    cv2.imwrite(folder_output + 'average_feature.png', generated_image)

    tools_image.plot_images(folder_output, '*.png')

    return
# ----------------------------------------------------------------------------------------------------------------------
def example_mnist():
    folder_features = 'data/features-mnist/FC/'
    folder_images = 'data/ex-mnist/'
    mask = '*.png'
    folder_output = 'data/output/'
    filename_model = 'data/ex42/GAN_mnist.h5'
    resize_H, resize_W = 32,32
    grayscale = True

    do_learn(folder_features,folder_images,folder_output,filename_model,resize_W=resize_W, resize_H = resize_H,grayscale=grayscale)
    do_generate(folder_features,filename_model,folder_output,           resize_W=resize_W, resize_H = resize_H,grayscale=grayscale)

    return
# ----------------------------------------------------------------------------------------------------------------------
def example_natural():

    folder_features = 'data/features-natural/CNN_App_Keras/'
    filename_model = 'data/Gen_MobileNet_model.h5'

    folder_images = 'data/ex-natural/'
    folder_output = 'data/output/'

    resize_H, resize_W = 16, 16
    grayscale = False

    do_learn(folder_features,folder_images,folder_output,filename_model, resize_W=resize_W, resize_H = resize_H,grayscale=grayscale)
    do_generate(folder_features,filename_model,folder_output,            resize_W=resize_W, resize_H = resize_H,grayscale=grayscale)

    return
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    #example_mnist()
    example_natural()