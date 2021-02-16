import numpy
import seaborn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
import classifier_KNN
import tools_DF
# ----------------------------------------------------------------------------------------------------------------------
def ex_01_ugly():
    df0 = seaborn.load_dataset('titanic')
    target = 'survived'
    c1 = 'sex'
    c2 = 'age'

    df = df0[[target, c1, c2]]
    df = df.dropna()
    df = tools_DF.hash_categoricals(df)


    seaborn.jointplot(data=df, x=c1, y=c2, hue=target, kind="kde", fill=True)
    plt.show()
    return
# ----------------------------------------------------------------------------------------------------------------------
def plot_density(X0,X1,confidence_mat,xx,yy,xlabel=None,ylabel=None):

    plt.contourf(xx, yy, numpy.flip(confidence_mat, 0), cmap=cm.coolwarm, alpha=.8)

    plt.plot(X0[:, 0], X0[:, 1], 'ro', color='blue', alpha=0.4)
    plt.plot(X1[: ,0], X1[:, 1], 'ro' ,color='red' , alpha=0.4)
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.savefig(folder_out+'density.png')

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_density():
    df = seaborn.load_dataset('titanic')
    df = df.dropna()
    df = tools_DF.hash_categoricals(df)
    target,c1, c2 = 'survived','age','sex'
    #target, c1, c2 = 'survived', 'age', 'pclass'

    X = df.loc[:, [c1, c2]].to_numpy()
    Y = df.loc[:, [target]].to_numpy().flatten()
    X0 = X[Y <= 0]
    X1 = X[Y > 0]

    Classifier = classifier_KNN.classifier_KNN()

    Classifier.learn(X, Y)

    minx, maxx = numpy.nanmin(X[:, 0]), numpy.nanmax(X[:, 0])
    miny, maxy = numpy.nanmin(X[:, 1]), numpy.nanmax(X[:, 1])
    xx, yy = numpy.meshgrid(numpy.linspace(minx, maxx, num=100), numpy.linspace(miny, maxy, num=100))
    confidence_mat = numpy.array(
        [(100 * Classifier.predict(x)[:, 1]).astype(int) for x in numpy.c_[xx.flatten(), yy.flatten()]])
    confidence_mat = confidence_mat.reshape((100, 100))

    plot_density(X0, X1, confidence_mat, xx, yy, c1, c2)
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    ex_density()





