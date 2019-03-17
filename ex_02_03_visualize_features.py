# ----------------------------------------------------------------------------------------------------------------------
import os
import cv2
import matplotlib.pyplot as plt
import fnmatch
from os import listdir
from sklearn import decomposition
import numpy
from sklearn.manifold import TSNE
# ----------------------------------------------------------------------------------------------------------------------
import tools_CNN_view
import tools_IO
import tools_ML
import tools_image
import tools_animation
# ----------------------------------------------------------------------------------------------------------------------
def plot_images_PCA(path_input,mask,resize_W,resize_H,grayscaled=True):
    patterns = numpy.sort(numpy.array([f.path[len(path_input):] for f in os.scandir(path_input) if f.is_dir()]))

    X,Y =[],[]
    for i in range(0, len(patterns)):
        local_images, labels, filenames = tools_IO.load_aligned_images_from_folder(path_input + patterns[i] + '/', patterns[i],mask=mask,grayscaled=grayscaled,resize_W=resize_W,resize_H=resize_H)
        local_images = local_images.reshape((local_images.shape[0],-1))
        X.append(local_images)
        Y.append(numpy.full(labels.shape[0],i))

    X = numpy.concatenate(X,axis=0)
    Y = numpy.concatenate(Y,axis=0).astype(numpy.int)

    X_PC = decomposition.PCA(n_components=2).fit_transform(X)
    X_TSNE = TSNE(n_components=2).fit_transform(X)

    fig = plt.figure(figsize=(12, 6))
    fig.subplots_adjust(hspace=0.01)

    tools_IO.plot_2D_scores_multi_Y(plt.subplot(1, 2, 1), X_TSNE, Y, labels=patterns)
    tools_IO.plot_2D_scores_multi_Y(plt.subplot(1, 2, 2), X_PC, Y)

    plt.tight_layout()
    plt.show()

    return
# ----------------------------------------------------------------------------------------------------------------------
def plot_features_PCA(path_input):


    patterns = fnmatch.filter(listdir(path_input),'*.txt')
    for i in range (0,len(patterns)):
        patterns[i]=patterns[i].split('.')[0]

    patterns = numpy.array(patterns)
    ML = tools_ML.tools_ML(None)

    X,Y,  filenames = ML.prepare_arrays_from_feature_files(path_input,patterns=patterns,feature_mask='.txt')
    X_PC = decomposition.PCA(n_components=2).fit_transform(X)
    X_TSNE = TSNE(n_components=2).fit_transform(X)

    fig = plt.figure(figsize=(12, 6))
    fig.subplots_adjust(hspace =0.01)

    tools_IO.plot_2D_scores_multi_Y(plt.subplot(1,2,1),X_TSNE, Y, labels=patterns)
    tools_IO.plot_2D_scores_multi_Y(plt.subplot(1,2,2),X_PC,   Y)

    plt.tight_layout()
    plt.show()

    return
# ----------------------------------------------------------------------------------------------------------------------
def save_features_as_tensors(folder_features,folder_output):
    patterns = numpy.array([each.split('.')[0] for each in fnmatch.filter(listdir(folder_features), '*.txt')])

    tools_IO.remove_files(folder_output)
    tools_IO.remove_folders(folder_output)
    for each in patterns: os.makedirs(folder_output + each)

    for p in range(0, len(patterns)):
        pattern = patterns[p]
        features, Y, filenames = tools_ML.tools_ML(None).prepare_arrays_from_feature_files(folder_features,patterns=numpy.array([pattern]),feature_mask='.txt')
        features = features[:, 1:].astype(numpy.float32)
        features*= 255.0/features.max()
        images = []
        for i in range (0,Y.shape[0]):
            image = tools_CNN_view.tensor_gray_1D_to_image(features[i], 'portrait').astype(numpy.uint8)
            cv2.imwrite(folder_output + pattern + '/' + filenames[i].split('.')[0]+'.png', tools_image.hitmap2d_to_viridis(image))
            images.append(image)

        image_avg = numpy.average(numpy.array(images), axis=0)
        image_avg = tools_image.hitmap2d_to_viridis(image_avg)
        cv2.imwrite(folder_output + pattern + '_avg.png',image_avg)

    tools_image.plot_images(folder_output, '*.png')

    return
# ----------------------------------------------------------------------------------------------------------------------
def average_images_from_subfolders(path_input,path_output,mask,resize_W=8, resize_H = 8,grayscaled=False):

    tools_IO.remove_files(path_output,create=True)
    patterns = numpy.sort(numpy.array([f.path[len(path_input):] for f in os.scandir(path_input) if f.is_dir()]))

    for i in range(0, len(patterns)):
        local_images, labels, filenames = tools_IO.load_aligned_images_from_folder(path_input + patterns[i] + '/', patterns[i],mask=mask,grayscaled=grayscaled,resize_W=resize_W,resize_H=resize_H)
        image_avg = numpy.zeros(local_images[0].shape)

        if grayscaled==False:
            image_avg[:, :, 0] = numpy.average(local_images[:, :, :, 0], axis=0)
            image_avg[:, :, 1] = numpy.average(local_images[:, :, :, 1], axis=0)
            image_avg[:, :, 2] = numpy.average(local_images[:, :, :, 2], axis=0)
        else:
            image_avg[:, :] = numpy.average(local_images, axis=0)

        image_avg*=255/image_avg.max()
        cv2.imwrite(path_output + patterns[i] + '_avg.png', image_avg)

    tools_image.plot_images(path_output, '*.png')

    return
# ----------------------------------------------------------------------------------------------------------------------
def example_natural():
    path_input_features = 'data/features-mnist/CNN_AlexNet_TF/'
    path_input_images,mask = 'data/ex09-mnist/','*.png'
    path_output = 'data/output/'

    #average_images_from_subfolders('data/output_gen_FC/','data/output/','*.jpg',resize_H=64,resize_W=64)
    #plot_features_PCA(path_input_features)
    #save_features_as_tensors(path_input_features,path_output)
    plot_images_PCA(path_input_images,mask,resize_W=8,resize_H=8,grayscaled=True)

    return
# ----------------------------------------------------------------------------------------------------------------------
def example_animation():
    tools_animation.folder_to_animated_gif_imageio('data/output_gen_FC/','data/output3/'    ,mask='.jpg',framerate=1,resize_H=64,resize_W=64)
    #tools_animation.folders_to_animated_gif('data/ex-natural/','data/output3/',mask='.jpg',framerate=3,resize_H=100,resize_W=100)

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    example_natural()
    #example_animation()