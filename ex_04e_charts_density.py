import numpy
import seaborn
import sklearn.datasets
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import tools_plot
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
import tools_ML_enhanced
import classifier_KNN
# ----------------------------------------------------------------------------------------------------------------------
def ex_01_ugly():
    df = seaborn.load_dataset('titanic')
    seaborn.jointplot(data=df, x="age", y="parch", hue="sex", kind="kde", fill=True)
    plt.show()
    return
# ----------------------------------------------------------------------------------------------------------------------
def plot_density(X0,X1,confidence_mat):
    minx, maxx = min(numpy.nanmin(X0[:,0]),numpy.nanmin(X1[:,0])),max(numpy.nanmax(X0[:,0]),numpy.nanmax(X1[:,0]))
    miny, maxy = min(numpy.nanmin(X0[:,1]),numpy.nanmin(X1[:,1])),max(numpy.nanmax(X0[:,1]),numpy.nanmax(X1[:,1]))
    S = 10

    xx, yy = numpy.meshgrid(numpy.linspace(minx, maxx, num=S), numpy.linspace(miny, maxy, num=S))
    plt.contourf(xx, yy, numpy.flip(confidence_mat, 0), cmap=cm.coolwarm, alpha=.8)

    plt.plot(X0[:, 0], X0[:, 1], 'ro', color='blue', alpha=0.4)
    plt.plot(X1[: ,0], X1[:, 1], 'ro' ,color='red' , alpha=0.4)
    plt.grid()
    plt.savefig(folder_out+'density.png')

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    df = seaborn.load_dataset('titanic')
    target,c1, c2 = 'survived','age','fare'

    X = df.loc[:,[c1,c2]].to_numpy()
    Y = df.loc[:,[target]].to_numpy().flatten()
    X0 = X[Y<=0]
    X1 = X[Y>0]
    confidence_mat = None

    Classifier = classifier_KNN.KNeighborsClassifier()
    ML = tools_ML_enhanced.tools_ML_enhanced(Classifier)
    plot_density(X0,X1,confidence_mat)




