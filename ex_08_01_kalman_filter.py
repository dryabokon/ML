#http://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/
# ----------------------------------------------------------------------------------------------------------------------
import numpy
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------
import detector_landmarks
import generator_TS
import tools_filter
import tools_IO
import tools_plot
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
    tools_IO.save_mat(tools_filter.do_filter_kalman_1D(S1, 0.2), filename4)

    tools_plot.plot_multiple_series(filename0, [filename1, filename2, filename3, filename4])
    plt.show()
    return
# ----------------------------------------------------------------------------------------------------------------------
def example_1D_02(X):
    Y = tools_filter.do_filter_kalman_1D(X, noise_level=1, Q=0.001)
    times = range(len(X))
    plt.plot(times, X, 'bo',times, Y[:, 0], 'b--')
    plt.show()
    return
# ----------------------------------------------------------------------------------------------------------------------
def example_2D(X):

    plt.figure()
    Y = tools_filter.do_filter_kalman_2D(X)
    times = range(X.shape[0])
    #plt.plot(times, X[:, 0], 'bo',times, X[:, 1], 'ro',times, Y[:, 0], 'b--',times, Y[:, 1], 'r--', )
    plt.plot(X[:, 0], X[:, 1])
    plt.plot(Y[:, 0], Y[:, 1])
    plt.show()
    return
# ----------------------------------------------------------------------------------------------------------------------
X = numpy.asarray([(399, 293), (403, 299), (409, 308), (416, 315), (418, 318), (420, 323), (429, 326), (423, 328), (429, 334),
(431, 337), (433, 342), (434, 352), (434, 349), (433, 350), (431, 350), (430, 349), (428, 347), (427, 345),
(425, 341), (429, 338), (431, 328), (410, 313), (406, 306), (402, 299), (397, 291), (391, 294), (376, 270),
(372, 272), (351, 248), (336, 244), (327, 236), (307, 220)])
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    data = tools_IO.load_mat('./data/ex_kalman/annotation.txt',delim=' ')[1:]
    filenames = numpy.array(data[:,0],dtype=numpy.str)
    P1s = numpy.array(data[:,[1,2]],dtype=int)
    P2s = numpy.array(data[:,[3,4]],dtype=int)
    IDs = numpy.array(data[:, 5],dtype=int)
    idx = numpy.where(IDs==0)

    ID = 7
    X = P1s[IDs==ID,0]
    Y = P1s[IDs==ID,1]

    X_filtered = tools_filter.do_filter_kalman_1D(X)
    Y_filtered = tools_filter.do_filter_kalman_1D(Y)


    times = range(len(X))
    plt.plot(times, X, 'bo',times,X_filtered,'b-',times, Y, 'ro',times,Y_filtered,'r-')
    plt.show()





