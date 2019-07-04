import numpy
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
import tools_ML
# ----------------------------------------------------------------------------------------------------------------------
import classifier_SVM
import classifier_RF
import classifier_Ada
import classifier_Gauss
import classifier_Bayes
import classifier_LM
import classifier_KNN
import classifier_Bayes2
import classifier_Gauss2
import classifier_Gauss_indep
# ---------------------------------------------------------------------------------------------------------------------
import tools_IO
# ----------------------------------------------------------------------------------------------------------------------
def benchmark_classifiers_ROC(filename_data_pos,filename_data_neg,path_out):

    filename_scrs_pos = path_out+'scores_pos.txt'
    filename_scrs_neg = path_out+'scores_neg.txt'

    TP, FP, AUC, DESC, = [],[],[],[]

    C1 = classifier_Ada.classifier_Ada()
    C2 = classifier_SVM.classifier_SVM()
    C3 = classifier_RF.classifier_RF()
    C4 = classifier_LM.classifier_LM()
    C5 = classifier_Bayes.classifier_Bayes()
    C6 = classifier_Bayes2.classifier_Bayes2()
    C7 = classifier_Gauss.classifier_Gauss()
    C8 = classifier_Gauss2.classifier_Gauss2()
    C9 = classifier_Gauss_indep.classifier_Gauss_indep()
    C10 = classifier_KNN.classifier_KNN()


    Classifiers = [C1, C2, C3, C4, C5, C6, C7, C8, C9, C10]

    for each in Classifiers:
        ML = tools_ML.tools_ML(each)

        tpr, fpr, auc = ML.E2E_features_2_classes(path_out,filename_data_pos, filename_data_neg,filename_scrs_pos,filename_scrs_neg)

        AUC.append(auc)
        TP.append(tpr)
        FP.append(fpr)
        DESC.append(each.name)

    idx = numpy.argsort(-numpy.array(AUC))

    TP = numpy.array(TP)[idx]
    FP = numpy.array(FP)[idx]
    AUC = numpy.array(AUC)[idx]
    DESC = numpy.array(DESC)[idx]

    tools_IO.plot_multiple_tp_fp(TP, FP, AUC, DESC)
    plt.show()

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    path_in  = 'data/ex_features_LPR/'
    path_out = 'data/output/'
    filename_data_pos = path_in + 'pos.txt'
    filename_data_neg = path_in + 'neg.txt'

    benchmark_classifiers_ROC(filename_data_pos,filename_data_neg,path_out)
