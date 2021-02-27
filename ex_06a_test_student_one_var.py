#https://samoedd.com/soft/r-t-test
# ----------------------------------------------------------------------------------------------------------------------
import numpy
import matplotlib.pyplot as plt
from scipy.stats import norm, ttest_ind, ttest_1samp, t
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
    plt.savefig(folder_out+'T_stat_one_var.png')
    return
# ----------------------------------------------------------------------------------------------------------------------
def is_fit_by_p_value(X,expexted_mean,a_significance_level=0.05,deg_of_freedom = None):

    t_stat, p_value = ttest_1samp(X, expexted_mean)
    if deg_of_freedom is None:
        deg_of_freedom = len(X) - 1


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
def is_fit_by_critical_value(X,expexted_mean,a_significance_level=0.05,deg_of_freedom = None):

    t_stat, p_value = ttest_1samp(X, expexted_mean)
    if deg_of_freedom is None:
        deg_of_freedom = len(X) - 1
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
def analyze_experiment(X_obs,expexted_mean):

    deg_of_freedom = len(X_obs) - 1
    a_significance_level = 0.05

    stat_value, p_value = ttest_1samp(X_obs, expexted_mean)
    is_fit_by_critical_value(X_obs, expexted_mean,a_significance_level)
    is_fit_by_p_value(X_obs, expexted_mean,a_significance_level)
    plot_t_stats(stat_value, deg_of_freedom, a_significance_level)

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_coin():
    X_obs = numpy.array([0, 1, 1, 1, 1, 1, 1, 0])
    analyze_experiment(X_obs, 0.5)
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    ex_coin()
