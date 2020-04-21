import numpy
import matplotlib.pyplot as plt
import warnings
import tools_ML_enhanced
import tools_plot
import tools_IO
warnings.filterwarnings('ignore')
# ----------------------------------------------------------------------------------------------------------------------
import classifier_RF
import classifier_Hash
import classifier_XGBoost
import classifier_XGBoost2
import classifier_DTree
import classifier_Bayes
import classifier_KNN
import classifier_SVM
import classifier_LM
# ---------------------------------------------------------------------------------------------------------------------
#Classifier = classifier_XGBoost.classifier_XGBoost()
Classifier = classifier_XGBoost2.classifier_XGBoost2()
#Classifier = classifier_Bayes.classifier_Bayes()
#Classifier = classifier_Hash.classifier_Hash()
#Classifier = classifier_KNN.classifier_KNN()
#Classifier = classifier_LM.classifier_LM()
#Classifier = classifier_SVM.classifier_SVM()
#Classifier = classifier_DTree.classifier_DT()
#Classifier = classifier_RF.classifier_RF()
# ---------------------------------------------------------------------------------------------------------------------
ML = tools_ML_enhanced.tools_ML_enhanced(Classifier)
# ---------------------------------------------------------------------------------------------------------------------
folder_out = './data/output/'
# ---------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_pocker/'
filename_in    = folder_in + 'train_formated.txt'
filename_train = folder_in + 'train_part1.txt'
filename_val   = folder_in + 'train_part2.txt'
filename_test  = folder_in + 'test_formated.txt'
# ---------------------------------------------------------------------------------------------------------------------
#filename_train = folder_in + 'train_enriched.txt'
#filename_val   = folder_in + 'val_enriched.txt'
# ---------------------------------------------------------------------------------------------------------------------
def preprocessing():
    idx = [26,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    tools_IO.switch_comumns(folder_in+'train.csv',folder_out+'train_formated.csv',idx=idx,delim=',')
    idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    tools_IO.switch_comumns(folder_in+'test.csv',folder_out+'test_formated.csv',idx=idx,delim=',')
    return
# ---------------------------------------------------------------------------------------------------------------------
def do_feature_importance():
    ML.split_train_test(filename_in, filename_train, filename_val, ratio=0.50, has_header=False, random_split=True)
    data = numpy.array(tools_IO.get_lines(filename_train), dtype=numpy.int)
    tools_plot.plot_feature_importance(plt, plt.figure(), data[:, 1:], data[:, 0],numpy.arange(0, data.shape[1] - 1, 1), folder_out + 'FI.png')
    return
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    ML.split_train_test(filename_in, filename_train, filename_val, ratio=0.50, has_header=False, random_split=True,max_line=None)
    ML.E2E_train_test(filename_train,filename_val,has_header=False,has_labels_first_col=True)

