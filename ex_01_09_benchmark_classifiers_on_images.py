import os
# ----------------------------------------------------------------------------------------------------------------------
import classifier_SVM
import classifier_RF
import classifier_Ada
import classifier_Gauss2
import classifier_Gauss_indep
import classifier_Bayes
import classifier_Bayes2
import classifier_LM
import classifier_KNN
# ---------------------------------------------------------------------------------------------------------------------
import tools_ML
import tools_IO
# ---------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
def benchmark_classifiers_on_images(path_input,path_output,mask,resize_H,resize_W,grayscaled, use_cashed_predictions=True):


    C1 = classifier_SVM.classifier_SVM()
    C2 = classifier_RF.classifier_RF()
    C3 = classifier_LM.classifier_LM()
    C4 = classifier_KNN.classifier_KNN()


    Classifiers = [C1, C2, C3, C4]


    for i in range(0,len(Classifiers)):
        folder_predictions = path_output + 'predictions/' + Classifiers[i].name + '/'
        filename_predictions = folder_predictions + Classifiers[i].name + '_predictions.txt'
        if use_cashed_predictions == False or not os.path.isfile(filename_predictions):
            ML = tools_ML.tools_ML(Classifiers[i])
            ML.E2E_images(path_input, folder_predictions, mask = mask, resize_W=resize_W, resize_H =resize_H,grayscaled = grayscaled,verbose=False)

    fig = plt.figure(figsize=(12, 6))
    fig.subplots_adjust(hspace =0.01)

    for i in range(0, len(Classifiers)):
        folder_predictions = path_output + 'predictions/' + Classifiers[i].name + '/'
        filename_predictions = folder_predictions + Classifiers[i].name + '_predictions.txt'
        if os.path.isfile(filename_predictions):
            tools_IO.plot_confusion_mat(plt.subplot(2, 2, i + 1), fig, filename_mat=filename_predictions, caption=Classifiers[i].name)

    plt.tight_layout()
    plt.show()

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    path_output = 'data/output/'

    path_input,mask = 'data/ex09-mnist/','*.png'
    resize_W, resize_H = 8,8
    grayscaled = True

    #path_input,mask = 'data/ex09-natural/','*.jpg'
    #resize_W, resize_H = 64,64
    #grayscaled = False



    benchmark_classifiers_on_images(path_input,path_output,mask,resize_H,resize_W,grayscaled=grayscaled,use_cashed_predictions=False)
