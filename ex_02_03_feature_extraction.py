# ----------------------------------------------------------------------------------------------------------------------
import CNN_AlexNet_TF
import CNN_Inception_TF
import CNN_App_Keras
# ----------------------------------------------------------------------------------------------------------------------
import classifier_FC_Keras
# ----------------------------------------------------------------------------------------------------------------------
def example_feature_extraction(path_input, path_output, mask,filename_model=None):

	CNN = CNN_AlexNet_TF.CNN_AlexNet_TF()
	#CNN = CNN_Inception_TF.CNN_Inception_TF()
	#CNN = CNN_App_Keras.CNN_App_Keras()
	#CNN = classifier_FC_Keras.classifier_FC_Keras(filename_weights=filename_model)

	CNN.generate_features(path_input, path_output+CNN.name+'/',mask=mask,limit=100)


	return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

	path_input  = 'data/ex09-mnist/'
	path_output = 'data/features-mnist/'
	filename_model = 'data/ex23/FC_model_dummy.h5'
	mask = '*.png'


	example_feature_extraction(path_input,path_output,mask,filename_model)