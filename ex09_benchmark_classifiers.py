import numpy
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
import tools_ML
# ----------------------------------------------------------------------------------------------------------------------
import classifier_SVM
import classifier_RF
import classifier_Ada
import classifier_Gauss
import classifier_Gauss2
import classifier_Gauss_indep
import classifier_Bayes
import classifier_Bayes2
import classifier_LM
import classifier_KNN
import classifier_FC
# ---------------------------------------------------------------------------------------------------------------------
import tools_IO
# ----------------------------------------------------------------------------------------------------------------------
def benchmark_classifiers_grid(filename_data_pos,filename_data_neg,path_out,noice_needed=0):

    filename_scrs_pos = path_out+'scores_pos.txt'
    filename_scrs_neg = path_out+'scores_neg.txt'
    filename_data_grid = path_out+'data_grid.txt'
    filename_scores_grid = path_out+'scores_grid.txt'

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
    C10 = classifier_FC.classifier_FC()



    Classifiers = [C1, C2, C3, C4, C5, C6, C7, C8, C9, C10]

    fig = plt.figure(figsize=(12, 6))
    fig.subplots_adjust(hspace =0.01)

    Pos = (tools_IO.load_mat(filename_data_pos, numpy.chararray, '\t')).shape[0]
    Neg = (tools_IO.load_mat(filename_data_neg, numpy.chararray, '\t')).shape[0]


    numpy.random.seed(125)
    idx_pos_train = numpy.random.choice(Pos, int(Pos/2),replace=False)
    idx_neg_train = numpy.random.choice(Neg, int(Neg/2),replace=False)
    idx_pos_test = [x for x in range(0, Pos) if x not in idx_pos_train]
    idx_neg_test = [x for x in range(0, Neg) if x not in idx_neg_train]


    for i in range(0,len(Classifiers)):
        ML = tools_ML.tools_ML(Classifiers[i])
        if i==0:ML.generate_data_grid(filename_data_pos, filename_data_neg, filename_data_grid)

        ML.learn_on_pos_neg_files(filename_data_pos, filename_data_neg, '\t', idx_pos_train,idx_neg_train)
        ML.score_feature_file(filename_data_pos, filename_scrs=filename_scrs_pos, delimeter='\t', append=0,rand_sel=idx_pos_test)
        ML.score_feature_file(filename_data_neg, filename_scrs=filename_scrs_neg, delimeter='\t', append=0,rand_sel=idx_neg_test)
        ML.score_feature_file(filename_data_grid, filename_scrs=filename_scores_grid)

        ML.learn_on_pos_neg_files(filename_data_pos, filename_data_neg, '\t', idx_pos_test,idx_neg_test)
        ML.score_feature_file(filename_data_pos, filename_scrs=filename_scrs_pos, delimeter='\t', append=1,rand_sel=idx_pos_train)
        ML.score_feature_file(filename_data_neg, filename_scrs=filename_scrs_neg, delimeter='\t', append=1,rand_sel=idx_neg_train)

        ML.score_feature_file(filename_data_grid, filename_scrs=filename_scores_grid)
        th = ML.get_th(filename_scrs_pos, filename_scrs_neg)
        tpr, fpr, roc_auc = tools_IO.get_roc_data_from_scores_file_v2(filename_scrs_pos, filename_scrs_neg)
        tools_IO.plot_2D_scores(plt.subplot(2, 5, i+1), fig, filename_data_pos, filename_data_neg, filename_data_grid,filename_scores_grid, th, noice_needed=noice_needed, caption=Classifiers[i].name + ' %1.2f'%roc_auc)


    plt.tight_layout()
    plt.show()

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    path_in  = 'data/ex09/'
    path_out = 'data/output/'
    filename_data_pos = path_in + 'data_pos.txt'
    filename_data_neg = path_in + 'data_neg.txt'

    benchmark_classifiers_grid(filename_data_pos,filename_data_neg,path_out,noice_needed=1)