# ----------------------------------------------------------------------------------------------------------------------
import numpy
from numpy.random import randn
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
# ----------------------------------------------------------------------------------------------------------------------
def do_filter(X,noise_level = 1,Q = 0.001):

    fk = KalmanFilter(dim_x=2, dim_z=1)
    fk.x = numpy.array([0., 1.])  # state (x and dx)
    fk.F = numpy.array([[1., 1.], [0., 1.]])

    fk.H = numpy.array([[1., 0.]])  # Measurement function
    fk.P = 10.  # covariance matrix
    fk.R = noise_level  # state uncertainty
    fk.Q = Q  # process uncertainty

    X_fildered, cov, _, _ = fk.batch_filter(X)

    return X_fildered[:,0]
# ----------------------------------------------------------------------------------------------------------------------
def plot_rts(noise_level):

    X = numpy.asarray([t + randn() * noise_level for t in range(40)])

    X_fildered = do_filter(X,noise_level)

    plt.plot(X,marker = 'o',ls = '', c = 'black',label = 'original')
    plt.plot(X_fildered, c='g', ls='--', label='KF output')

    plt.legend(loc=4)
    plt.grid()
    plt.show()
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    plot_rts(7.)