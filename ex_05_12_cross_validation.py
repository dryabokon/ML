import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    clf = svm.SVC(kernel='linear', C=1)
    iris = datasets.load_iris()
    scores = cross_val_score(clf, iris.data, iris.target)

    print(scores)