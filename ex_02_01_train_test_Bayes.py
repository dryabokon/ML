import os
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
import tools_ML
import tools_IO
# ----------------------------------------------------------------------------------------------------------------------
import classifier_Bayes
import classifier_SVM
import classifier_RF
import classifier_Ada
import classifier_Gauss
import classifier_LM
import classifier_KNN
# ---------------------------------------------------------------------------------------------------------------------
import generator_Bay
import generator_Gauss
import generator_Other
import generator_Manual

# ---------------------------------------------------------------------------------------------------------------------
def generate_data_syntetic(folder_output,dim = 2):
    if not os.path.exists(folder_output):
        os.makedirs(folder_output)
    else:
        tools_IO.remove_files(folder_output)

    Generator = generator_Bay.generator_Bay(dim)
    #Generator = generator_Gauss.generator_Gauss(dim)
    #Generator = generator_Other.generator_Other(dim)

    Generator.create_pos_neg_samples(folder_output + 'data_pos.txt', folder_output + 'data_neg.txt')
    tools_IO.plot_2D_samples_from_folder(folder_output,  add_noice=1)
    plt.show()
    return 0
# ---------------------------------------------------------------------------------------------------------------------
def classify_data(filename_data_pos,filename_data_neg,folder_out):

    filename_scrs_pos = folder_out+'scores_pos.txt'
    filename_scrs_neg = folder_out+'scores_neg.txt'

    Classifier = classifier_Bayes.classifier_Bayes()
    #Classifier = classifier_KNN.classifier_KNN()
    #Classifier = classifier_Ada.classifier_Ada()
    #Classifier = classifier_SVM.classifier_SVM()
    #Classifier = classifier_RF.classifier_RF()
    #Classifier = classifier_LM.classifier_LM()
    #Classifier = classifier_Gauss.classifier_Gauss()


    fig = plt.figure(figsize=(12, 6))
    fig.subplots_adjust(hspace=0.01)
    ML = tools_ML.tools_ML(Classifier)
    ML.E2E_features_2_classes(folder_out, filename_data_pos, filename_data_neg,filename_scrs_pos=filename_scrs_pos, filename_scrs_neg=filename_scrs_neg,fig=fig)

    return 0
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    folder_in = 'data/ex_pos_neg_bayes/'
    folder_out = 'data/output/'
    classify_data(folder_in+'pos.txt', folder_in+'neg.txt',folder_out)

