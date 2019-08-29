from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
# ----------------------------------------------------------------------------------------------------------------------
import numpy
import tools_IO
import tools_ML
# ----------------------------------------------------------------------------------------------------------------------
def example_01():
    clf = svm.SVC(kernel='linear', C=1)
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    scores = cross_val_score(clf, X, Y)
    print(scores)
    return
# ----------------------------------------------------------------------------------------------------------------------
def example_02(path_input,has_header,has_labels_first_col):
    clf = svm.SVC(kernel='linear', C=1)


    patterns = []
    filenames = tools_IO.get_filenames(path_input,'*.txt')
    for filename in filenames:
        patterns.append(filename.split('.txt')[0])
    patterns = numpy.array(patterns)

    ML = tools_ML.tools_ML(None)

    (X, Y, filenames) = ML.prepare_arrays_from_feature_files(path_input, patterns,has_header=has_header,has_labels_first_col=has_labels_first_col)
    scores = cross_val_score(clf, X, Y)
    print(scores)

    return
# ----------------------------------------------------------------------------------------------------------------------
has_header,has_labels_first_col = True, True
folder_in = 'data/ex_pos_neg_apnea/'
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    example_02(folder_in,has_header,has_labels_first_col)