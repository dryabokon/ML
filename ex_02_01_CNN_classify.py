# ----------------------------------------------------------------------------------------------------------------------
import CNN_AlexNet_TF
import CNN_VGG16_Keras
import CNN_Inception_TF
import CNN_Resnet_TF_Hub
# ----------------------------------------------------------------------------------------------------------------------
def example_classification():

	path_input,mask = 'data/ex09-natural/','*.jpg'

	CNN = CNN_AlexNet_TF.CNN_AlexNet_TF()
	#CNN = CNN_VGG16_Keras.CNN_VGG16_Keras()
	#CNN = CNN_Inception_TF.CNN_Inception_TF()
	#CNN = CNN_Resnet_TF_Hub.CNN_Resnet_TF()

	CNN.predict_classes(path_input, 'descript.ion', mask=mask, limit=200)
	return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

	example_classification()