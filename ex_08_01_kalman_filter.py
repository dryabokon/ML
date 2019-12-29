# ----------------------------------------------------------------------------------------------------------------------
import numpy
from numpy.random import randn
import matplotlib.pyplot as plt
import tools_filter
import generator_TS
import tools_IO
import sklearn
# ----------------------------------------------------------------------------------------------------------------------
D = detector_landmarks.detector_landmarks('..//_weights//shape_predictor_68_face_landmarks.dat')
# ----------------------------------------------------------------------------------------------------------------------
def example_1D():
    filename0 = './data/output/original.txt'
    filename1 = './data/output/noised.txt'
    filename2 = './data/output/filter_mean.txt'
    filename3 = './data/output/filter_median.txt'
    filename4 = './data/output/filter_kalman.txt'

    G = generator_TS.generator_TS(2, 100)
    S0 = G.generate_sine(filename0, 15, 0.00)[:, 0]
    S1 = G.generate_sine(filename1, 15, 0.10)[:, 0]

    tools_IO.save_mat(tools_filter.do_filter_average(S1, 11), filename2)
    tools_IO.save_mat(tools_filter.do_filter_median(S1, 11), filename3)
    tools_IO.save_mat(tools_filter.do_filter_kalman(S1, 0.2), filename4)

    tools_IO.plot_multiple_series(filename0, [filename1, filename2, filename3, filename4])
    plt.show()
    return
# ----------------------------------------------------------------------------------------------------------------------
def MAE(S1,S2):
    return sklearn.metrics.mean_absolute_error(S1,S2)
# ----------------------------------------------------------------------------------------------------------------------
def plot_2D_samples(S,labels):

    colors_list = list(('red', 'blue', 'green', 'orange', 'cyan', 'purple', 'black', 'gray', 'pink', 'darkblue'))

    for i in range(len(S)):
        label = labels[i]
        XY = S[i]
        plt.plot(XY[:, 0], XY[:, 1], 'ro', color=colors_list[i%len(colors_list)], alpha=0.4,lw=1, linestyle='--')

    plt.tight_layout()
    plt.grid()
    return
# ----------------------------------------------------------------------------------------------------------------------
def example_2D():
    filename0x = './data/output/original_x.txt'
    filename0y = './data/output/original_x.txt'

    G = generator_TS.generator_TS(2, 50)
    S0x = G.generate_sine(filename0x, 15, 0.00)[:, 0]
    S0y = G.generate_sine(filename0y, 13, 0.00)[:, 0]
    S0 = numpy.vstack((S0x, S0y)).T

    S1x = G.generate_sine(filename0x, 15, 0.05)[:, 0]
    S1y = G.generate_sine(filename0y, 13, 0.05)[:, 0]
    S1 = numpy.vstack((S1x, S1y)).T

    plot_2D_samples([S0,S1],['original','noise'])
    plt.show()
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    example_1D()
