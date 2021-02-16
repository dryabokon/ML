import numpy
import pandas as pd
import sklearn.datasets
import seaborn
from sklearn.datasets import fetch_openml
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
dataset_iris = sklearn.datasets.load_iris()
dataset_wine = sklearn.datasets.load_wine()
dataset_boston = sklearn.datasets.load_boston()
dataset_diabetes = sklearn.datasets.load_diabetes()
dataset_moons = sklearn.datasets.make_moons(n_samples=100)
dataset_circles = sklearn.datasets.make_circles(n_samples=100)
# ----------------------------------------------------------------------------------------------------------------------
dataset_linear = [None,None]
dataset_linear[0],dataset_linear[1] = sklearn.datasets.make_classification(n_features=2, n_redundant=0, n_informative=2,random_state=1, n_clusters_per_class=1)
dataset_linear[0] = numpy.array(dataset_linear[0]) + 2*numpy.random.uniform(size=dataset_linear[0].shape)
# ----------------------------------------------------------------------------------------------------------------------
def import_dataset_sklearn(sklearn_dataset):

    X = sklearn_dataset['data']
    Y = sklearn_dataset['target']
    feature_names = numpy.array(sklearn_dataset['feature_names'])
    df = pd.DataFrame(data=numpy.hstack((numpy.expand_dims(Y, axis=1), X)), columns=['target'] + feature_names.tolist())

    return df
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    T = numpy.array(dataset_moons[1]).reshape((-1, 1))
    X1 = numpy.array(dataset_moons[0][:, 0]).reshape((-1, 1))
    X2 = numpy.array(dataset_moons[0][:, 1]).reshape((-1, 1))
    df = pd.DataFrame(numpy.concatenate((T, X1, X2), axis=1))
    df.to_csv(folder_out + 'dataset_moons.txt', index=False, sep='\t')

    T = numpy.array(dataset_circles[1]).reshape((-1, 1))
    X1 = numpy.array(dataset_circles[0][:, 0]).reshape((-1, 1))
    X2 = numpy.array(dataset_circles[0][:, 1]).reshape((-1, 1))
    df = pd.DataFrame(numpy.concatenate((T, X1, X2), axis=1))
    df.to_csv(folder_out + 'dataset_circles.txt', index=False, sep='\t')

    T = numpy.array(dataset_linear[1]).reshape((-1, 1))
    X1 = numpy.array(dataset_linear[0][:, 0]).reshape((-1, 1))
    X2 = numpy.array(dataset_linear[0][:, 1]).reshape((-1, 1))
    df = pd.DataFrame(numpy.concatenate((T, X1, X2), axis=1))
    df.to_csv(folder_out + 'dataset_linear.txt', index=False, sep='\t')


    df = import_dataset_sklearn(dataset_wine)
    df.to_csv(folder_out + 'dataset_wine.txt', index=False, sep='\t')

    df = import_dataset_sklearn(dataset_diabetes)
    df.to_csv(folder_out + 'dataset_diabetes.txt', index=False, sep='\t')

    df = seaborn.load_dataset('tips')
    df.to_csv(folder_out + 'dataset_tips.txt', index=False, sep='\t')

    df = seaborn.load_dataset('flights')
    df.to_csv(folder_out + 'dataset_flights.txt', index=False, sep='\t')

    df = seaborn.load_dataset('exercise')
    df.to_csv(folder_out + 'dataset_exercise.txt', index=False, sep='\t')

    df = seaborn.load_dataset('titanic')
    df.to_csv(folder_out + 'dataset_titanic.txt', index=False, sep='\t')

    survey = fetch_openml(data_id=534,return_X_y=False)
    df = pd.DataFrame(data=numpy.hstack((numpy.expand_dims(survey.target, axis=1), survey.data)), columns=['target'] + survey.feature_names)
    df.to_csv(folder_out + 'dataset_survey.txt', index=False, sep='\t')

