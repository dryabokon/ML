import os
# ----------------------------------------------------------------------------------------------------------------------
import classifier_SVM
import classifier_RF
import classifier_Gauss2
import classifier_Bayes2
import classifier_LM
import classifier_KNN
# ---------------------------------------------------------------------------------------------------------------------
import tools_ML
import tools_IO
# ---------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
def benchmark_classifiers(path_features,path_output,has_header,has_labels_first_col):


    C1 = classifier_SVM.classifier_SVM()
    C2 = classifier_RF.classifier_RF()
    C3 = classifier_LM.classifier_LM()
    C4 = classifier_Bayes2.classifier_Bayes2()
    C5 = classifier_Gauss2.classifier_Gauss2()
    C6 = classifier_KNN.classifier_KNN()

    Classifiers = [C1, C2, C3, C4, C5, C6]


    for i in range(0,len(Classifiers)):
        folder_predictions = path_output + 'predictions/' + Classifiers[i].name + '/'
        ML = tools_ML.tools_ML(Classifiers[i])
        ML.E2E_features(path_features, folder_predictions,mask = '.txt',limit_classes=20, limit_instances=100,has_header=has_header,has_labels_first_col=has_labels_first_col)


    fig = plt.figure(figsize=(12, 6))
    fig.subplots_adjust(hspace =0.01)

    for i in range(0, len(Classifiers)):
        folder_predictions = path_output + 'predictions/' + Classifiers[i].name + '/'
        filename_predictions = folder_predictions + Classifiers[i].name + '_predictions.txt'
        if os.path.isfile(filename_predictions):
            tools_IO.plot_confusion_mat(plt.subplot(2, 3, i + 1), fig, filename_mat=filename_predictions, caption=Classifiers[i].name)

    plt.tight_layout()
    plt.show()

    return
# ----------------------------------------------------------------------------------------------------------------------
#has_header,has_labels_first_col = True, True
#path_in = 'data/ex_pos_neg_apnea/'
# ----------------------------------------------------------------------------------------------------------------------
has_header,has_labels_first_col = False, True
path_in = 'data/ex_features_digits_mnist/CNN_AlexNet_TF/'
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    path_output = 'data/output/'
    benchmark_classifiers(path_in,path_output,has_header,has_labels_first_col)
