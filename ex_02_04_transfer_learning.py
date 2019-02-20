# ----------------------------------------------------------------------------------------------------------------------
import classifier_SVM
import classifier_RF
import classifier_Ada
import classifier_XGBoost
import classifier_Gauss
import classifier_Gauss2
import classifier_Gauss_indep
import classifier_Bayes
import classifier_Bayes2
import classifier_LM
import classifier_KNN
import classifier_FC_Keras
# ----------------------------------------------------------------------------------------------------------------------
import tools_ML
# ----------------------------------------------------------------------------------------------------------------------
def example_train_test_on_features():
	path_input = 'data/features-natural/FC/'
	path_output = 'data/features-natural/'

	C0 = classifier_Ada.classifier_Ada()
	C1 = classifier_XGBoost.classifier_XGBoost()
	C2 = classifier_SVM.classifier_SVM()
	C3 = classifier_RF.classifier_RF()
	C4 = classifier_KNN.classifier_KNN()

	C5 = classifier_Bayes.classifier_Bayes()
	C6 = classifier_Bayes2.classifier_Bayes2()
	C7 = classifier_Gauss.classifier_Gauss()
	C8 = classifier_Gauss2.classifier_Gauss2()
	C9 = classifier_Gauss_indep.classifier_Gauss_indep()
	C10 = classifier_LM.classifier_LM()

	C11 = classifier_FC_Keras.classifier_FC_Keras()


	P = tools_ML.tools_ML(C11)
	P.E2E_features(path_input, path_output,  limit_classes=40, limit_instances=2000)


	return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

	example_train_test_on_features()








