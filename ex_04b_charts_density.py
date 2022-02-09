import numpy
import seaborn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.datasets import make_regression
# ----------------------------------------------------------------------------------------------------------------------
from classifier import classifier_LM
import tools_DF
import tools_plot_v2
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
P = tools_plot_v2.Plotter(folder_out)
# ----------------------------------------------------------------------------------------------------------------------
def ex_01_ugly():
    df0 = seaborn.load_dataset('titanic')
    target,c1,c2 = 'survived','sex', 'age'

    df = df0[[target, c1, c2]]
    df = df.dropna()
    df = tools_DF.hash_categoricals(df)

    seaborn.jointplot(data=df, x=c1, y=c2, hue=target, kind="kde", fill=True)
    plt.show()
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_density(X,Y):

    Classifier = classifier_LM.classifier_LM()
    Classifier.learn(X, Y)
    minx, maxx = numpy.nanmin(X[:, 0]), numpy.nanmax(X[:, 0])
    miny, maxy = numpy.nanmin(X[:, 1]), numpy.nanmax(X[:, 1])
    xx, yy = numpy.meshgrid(numpy.linspace(minx, maxx, num=100), numpy.linspace(miny, maxy, num=100))
    confidence_mat = numpy.array([(100 * Classifier.predict(x)[:, 1]).astype(int) for x in numpy.c_[xx.flatten(), yy.flatten()]])
    grid_confidence = confidence_mat.reshape((100, 100))
    P.plot_contourf(X[Y<=0], X[Y>0], xx, yy, grid_confidence, filename_out='density.png')
    return
# ----------------------------------------------------------------------------------------------------------------------
def get_data_regression():
    c1, c2 = 'c1', 'c2'
    X, Y = make_regression(n_samples=1000, n_features=2,noise=20.0)
    Y[Y<=0]=0
    Y[Y >0]=1
    return X,Y
# ----------------------------------------------------------------------------------------------------------------------
def get_data_titanic():
    df = seaborn.load_dataset('titanic')
    df = df.dropna()
    df = tools_DF.hash_categoricals(df)
    target,c1, c2 = 'survived','sex', 'deck'
    X = df.loc[:, [c1, c2]].to_numpy()
    Y = df.loc[:, [target]].to_numpy().flatten()
    return X,Y
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #X,Y = get_data_regression()
    X, Y = get_data_titanic()
    ex_density(X,Y)





