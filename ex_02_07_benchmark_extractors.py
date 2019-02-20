import os
# ----------------------------------------------------------------------------------------------------------------------
import classifier_RF
# ---------------------------------------------------------------------------------------------------------------------
import CNN_AlexNet_TF
import CNN_Inception_TF
#import CNN_VGG16_Keras
# ---------------------------------------------------------------------------------------------------------------------
import tools_ML
import tools_IO
# ---------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
def benchmark_extractors():

    mask = '*.jpg'
    path_input = 'data/ex-natural'
    path_output = 'data/features/'

    E1 = CNN_AlexNet_TF.CNN_AlexNet_TF()
    E2 = CNN_Inception_TF.CNN_Inception_TF()
    #E3 = CNN_VGG16_Keras.CNN_VGG16_Keras()

    Extractors = [E1,E2]
    Classifier = classifier_RF.classifier_RF()
    ML = tools_ML.tools_ML(Classifier)

    fig = plt.figure(figsize=(12, 6))
    fig.subplots_adjust(hspace=0.01)

    for i in range(0,len(Extractors)):
        path_features = path_output + Extractors[i].name
        if not os.path.exists(path_features):
            Extractors[i].generate_features(path_input, path_features, mask=mask, limit=200)

        folder_predictions = path_output + 'predictions/' + Extractors[i].name + '/'
        filename_predictions = folder_predictions + Classifier.name + '_predictions.txt'
        if not os.path.isfile(filename_predictions):
            ML.E2E_features(path_features, folder_predictions, limit_classes=20, limit_instances=100)

        tools_IO.plot_confusion_mat(plt.subplot(1, 3, i + 1), fig, filename_mat=filename_predictions,caption=Classifier.name+ ' + '+ Extractors[i].name)


    plt.tight_layout()
    plt.show()

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    benchmark_extractors()








