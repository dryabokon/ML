import numpy as numpy
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import multivariate_normal
# ----------------------------------------------------------------------------------------------------------------------
import tools_plot_v2
import tools_DF
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
P = tools_plot_v2.Plotter(folder_out)
# ----------------------------------------------------------------------------------------------------------------------
def get_data(fraction_outlier):

    mean      , cov       = [3, 2], [[10, 2], [2, 10]]
    mean_noise, cov_noise = [19, 15], [[6, 0], [0, 6]]

    x, y = numpy.random.multivariate_normal(mean, cov, 500).T
    X_main = numpy.concatenate((x.reshape(-1, 1), (y.reshape(-1, 1))), axis=1)

    x, y = numpy.random.multivariate_normal(mean_noise,cov_noise, int(X_main.shape[0]*fraction_outlier/(1-fraction_outlier))).T
    X_noise = numpy.concatenate((x.reshape(-1,1),(y.reshape(-1,1))),axis=1)

    X = numpy.concatenate((X_main,X_noise))
    Y = numpy.concatenate((-numpy.ones((X_main.shape[0])),+numpy.ones((X_noise.shape[0]))),axis=0)

    xx, yy = get_meshgrid(X)
    x1,x2 = xx.min(),xx.max()
    y1,y2 = yy.min(),yy.max()
    d = (x2-x2)*0.1

    confidence_mat = numpy.array([(10**8*multivariate_normal.pdf(x, mean, cov)).astype(int) for x in numpy.c_[xx.flatten(), yy.flatten()]])
    confidence_mat=confidence_mat/numpy.max(confidence_mat)
    grid_confidence = -numpy.log(confidence_mat+(1e-5)).reshape((100, 100))
    P.plot_contourf(X[Y <= 0], X[Y > 0], xx, yy, grid_confidence, x_range=[x1 - d, x2 + d], y_range=[y1 - d, y2 + d],filename_out='1_GT_density.png')
    P.plot_2D_features_multi_Y(X, Y,x_range=[x1 - d, x2 + d], y_range=[y1 - d, y2 + d], filename_out='1_GT.png')
    return X,Y
# ----------------------------------------------------------------------------------------------------------------------
def get_meshgrid(X):
    x1,x2 = numpy.nanmin(X[:, 0]), numpy.nanmax(X[:, 0])
    y1,y2 = numpy.nanmin(X[:, 1]), numpy.nanmax(X[:, 1])
    xx, yy = numpy.meshgrid(numpy.linspace(x1,x2, num=100),numpy.linspace(y1,y2, num=100))
    return xx, yy
# ----------------------------------------------------------------------------------------------------------------------
def normal_fit_predict(X, fraction_outlier):
    mean = X.mean(axis=0)
    cov = (X - mean).T @ (X - mean) / X.shape[0]
    p = multivariate_normal.pdf(X, mean, cov)
    idx = numpy.argsort(p)
    Y = numpy.ones(X.shape[0])
    Y[idx[:int(fraction_outlier * X.shape[0])]]=-1

    xx, yy = get_meshgrid(X)
    x1,x2 = xx.min(),xx.max()
    y1,y2 = yy.min(),yy.max()
    d = (x2-x2)*0.1

    confidence_mat = numpy.array([(10 ** 8 * multivariate_normal.pdf(x, mean, cov)).astype(int) for x in numpy.c_[xx.flatten(), yy.flatten()]])
    confidence_mat = confidence_mat / numpy.max(confidence_mat)
    grid_confidence = -numpy.log(confidence_mat + (1e-5)).reshape((100, 100))
    P.plot_contourf(X[Y > 0], X[Y <= 0], xx, yy, grid_confidence, x_range=[x1-d,x2+d],y_range=[y1-d,y2+d],filename_out='2_pred_Normal_density.png')
    P.plot_2D_features_multi_Y(X, -Y, x_range=[x1-d,x2+d],y_range=[y1-d,y2+d],filename_out='2_pred_Normal.png')
    return
# ----------------------------------------------------------------------------------------------------------------------
def predict_IsolationForest(X, fraction_outlier):
    xx, yy = get_meshgrid(X)
    x1, x2 = xx.min(), xx.max()
    y1, y2 = yy.min(), yy.max()
    d = (x2 - x2) * 0.1

    A = IsolationForest(contamination=fraction_outlier)
    A.fit(X)
    Y = A.predict(X)
    P.plot_2D_features(X, -Y,filename_out='3_pred_IsolationForest.png')

    confidence_mat = A.decision_function(numpy.c_[xx.ravel(), yy.ravel()])
    confidence_mat = confidence_mat / numpy.max(confidence_mat)
    grid_confidence = -numpy.log(confidence_mat + (1e-5)).reshape((100, 100))
    P.plot_contourf(X[Y > 0], X[Y <= 0], xx, yy, grid_confidence, x_range=[x1 - d, x2 + d], y_range=[y1 - d, y2 + d],filename_out='3_pred_IsolationForest_density.png')
    return
# ----------------------------------------------------------------------------------------------------------------------
def predict_EllipticEnvelope(X, fraction_outlier):
    xx, yy = get_meshgrid(X)
    x1, x2 = xx.min(), xx.max()
    y1, y2 = yy.min(), yy.max()
    d = (x2 - x2) * 0.1

    A = EllipticEnvelope(contamination=fraction_outlier)
    A.fit(X)
    Y = A.predict(X)

    confidence_mat = numpy.array([(A.predict(x.reshape(-1,2))).astype(int) for x in numpy.c_[xx.flatten(), yy.flatten()]])
    grid_confidence = (confidence_mat).reshape((100, 100))
    P.plot_contourf(X[Y > 0], X[Y <= 0], xx, yy, grid_confidence, x_range=[x1 - d, x2 + d], y_range=[y1 - d, y2 + d],filename_out='4_pred_EllipticEnvelope_density.png')
    P.plot_2D_features_multi_Y(X, -Y, x_range=[x1 - d, x2 + d], y_range=[y1 - d, y2 + d],filename_out='4_pred_EllipticEnvelope.png')
    return
# ----------------------------------------------------------------------------------------------------------------------
def predict_LocalOutlierFactor(X, fraction_outlier):
    xx, yy = get_meshgrid(X)
    x1, x2 = xx.min(), xx.max()
    y1, y2 = yy.min(), yy.max()
    d = (x2 - x2) * 0.1

    A = LocalOutlierFactor(contamination=fraction_outlier, novelty=True)
    A.fit(X)
    Y = A.predict(X)

    confidence_mat = numpy.array([(A.predict(x.reshape(-1,2))).astype(int) for x in numpy.c_[xx.flatten(), yy.flatten()]])
    grid_confidence = (confidence_mat).reshape((100, 100))
    P.plot_contourf(X[Y > 0], X[Y <= 0], xx, yy, grid_confidence, x_range=[x1 - d, x2 + d], y_range=[y1 - d, y2 + d],filename_out='5_pred_LocalOutlierFactor_density.png')
    P.plot_2D_features_multi_Y(X, -Y, x_range=[x1 - d, x2 + d], y_range=[y1 - d, y2 + d],filename_out='5_pred_LocalOutlierFactor.png')
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_simple():
    fraction_outlier = 0.05
    X, Y = get_data(fraction_outlier)
    P.plot_2D_features_multi_Y(X, -IsolationForest(contamination=fraction_outlier).fit_predict(X), filename_out='3_pred_IsolationForest.png')
    P.plot_2D_features_multi_Y(X, -EllipticEnvelope(contamination=fraction_outlier).fit_predict(X), filename_out='4_pred_EllipticEnvelope.png')
    P.plot_2D_features_multi_Y(X, -LocalOutlierFactor(contamination=fraction_outlier, novelty=True).fit(X).predict(X), filename_out='5_pred_LocalOutlierFactor.png')
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    fraction_outlier = 0.05
    X,Y = get_data(fraction_outlier)
    predict_IsolationForest(X, fraction_outlier)
    normal_fit_predict(X, fraction_outlier)
    predict_EllipticEnvelope(X, fraction_outlier)
    predict_LocalOutlierFactor(X, fraction_outlier)




