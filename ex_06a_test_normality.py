#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html
# ----------------------------------------------------------------------------------------------------------------------
import numpy
import matplotlib.pyplot as plt
from scipy.stats import normaltest, probplot, kstest
import statsmodels.api as sm
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
def ex_normality_test(X):

    value_stat, p_value = normaltest(X)
    #value_stat, p_value = kstest(X, 'norm')

    a_significance_level = 0.05

    if p_value <= a_significance_level:
        print("p value <= significance_level")
        print('%1.2f    < %1.2f' % (p_value, a_significance_level))
        print('Reject H0: normal distribution is NOT confirmed')
    else:
        print("significance_level < p value")
        print('%1.2f               < %1.2f' % (a_significance_level, p_value))
        print('Accept H0: normal distribution is confirmed')
    print()

    probplot(X, dist='norm', plot=plt)
    plt.savefig(folder_out+'qqplot.png')
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    N = 1000
    mean0,mean1 = 0,0
    std0,std1 = 2,2

    X = numpy.concatenate((numpy.random.normal(mean0, std0, size=N), numpy.random.normal(mean1, std1, size=N)))
    ex_normality_test(X)


