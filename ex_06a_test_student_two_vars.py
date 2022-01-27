#https://samoedd.com/soft/r-t-test
# ----------------------------------------------------------------------------------------------------------------------
import numpy
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, t
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
def plot_t_stats(stat_value,deg_of_freedom,a_significance_level=0.05):

    critical_value = t.ppf(1 - a_significance_level / 2, deg_of_freedom)
    X_good = numpy.linspace(0, critical_value, 50)
    X_bad = numpy.linspace(critical_value, critical_value * 2, 50)

    Y_good = t.pdf(X_good, deg_of_freedom)
    Y_bad = t.pdf(X_bad, deg_of_freedom)

    plt.plot(X_good, Y_good, color=(0, 0, 0.5))
    plt.plot(X_bad, Y_bad, color=(0.5, 0, 0))

    plt.fill_between(X_good, Y_good, color=(0.7, 0.8, 1))
    plt.fill_between(X_bad, Y_bad, color=(1, 0.9, 0.9))

    marker = 'bo' if stat_value<critical_value else 'ro'

    plt.plot(stat_value,0,marker)
    plt.tight_layout()
    plt.savefig(folder_out+'T_stat_two_vars.png')
    return
# ----------------------------------------------------------------------------------------------------------------------
def is_fit_by_p_value(X1_obs, X2_obs,a_significance_level=0.05):

    t_stat, p_value = ttest_ind(X1_obs, X2_obs, equal_var=False)

    if p_value <= a_significance_level:
        print("p value <= significance_level")
        print('%1.2f    < %1.2f' % (p_value, a_significance_level))
        print('Reject H0: mean value no fit')
    else:
        print("significance_level < p value")
        print('%1.2f               < %1.2f' % (a_significance_level, p_value))
        print('Accept H0: mean value fit')
    print()

    return
# ----------------------------------------------------------------------------------------------------------------------
def is_fit_by_critical_value(X1_obs, X2_obs,a_significance_level=0.05,deg_of_freedom = None):

    t_stat, p_value = ttest_ind(X1_obs, X2_obs, equal_var=False)
    if deg_of_freedom is None:
        m, n = len(X1_obs), len(X2_obs)
        sx2 = ((X1_obs - numpy.mean(X1_obs)) ** 2).sum() / (m - 1)
        sy2 = ((X2_obs - numpy.mean(X2_obs)) ** 2).sum() / (n - 1)
        s2 = sx2 / m + sy2 / n
        deg_of_freedom = s2 ** 2 / (sx2 ** 2 / ((m - 1) * m ** 2) + sy2 ** 2 / ((n - 1) * n ** 2))

    critical_value = t.ppf(1-a_significance_level/2, deg_of_freedom)

    if t_stat< critical_value:
        print('t_stat < critical_value')
        print('%1.2f            < %1.2f' % (t_stat, critical_value))
        print('Accept H0: fit OK')
    else:
        print("critical_value <= t_stat")
        print('%1.2f          <= %1.2f' % (critical_value, t_stat))
        print('Reject H0: no fit')

    print()

    return
# ----------------------------------------------------------------------------------------------------------------------
def analyze_experiment(X1_obs,X2_obs):

    a_significance_level = 0.05
    m,n = len(X1_obs),len(X2_obs)
    sx2 = ((X1_obs-numpy.mean(X1_obs))**2).sum()/(m-1)
    sy2 = ((X2_obs-numpy.mean(X2_obs))**2).sum()/(n-1)
    s2 = sx2/m + sy2/n
    deg_of_freedom0 = s2**2/(sx2**2/((m-1)*m**2)+sy2**2/((n-1)*n**2))
    deg_of_freedom = m+n-2

    stat_value, p_value = ttest_ind(X1_obs, X2_obs, equal_var=False)

    is_fit_by_p_value(X1_obs, X2_obs, a_significance_level)
    is_fit_by_critical_value(X1_obs, X2_obs,a_significance_level,deg_of_freedom)
    plot_t_stats(stat_value, deg_of_freedom, a_significance_level)

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_coin():
    X1_obs = numpy.array([0, 0, 1, 1, 1, 1, 1, 1])
    X2_obs = numpy.array([0, 0, 0, 1, 1, 1,0, 0, 0, 1, 1, 1,0, 0, 0, 1, 1, 1])
    analyze_experiment(X1_obs, X2_obs)
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    ex_coin()
