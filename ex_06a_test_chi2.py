#https://statanaliz.info/statistica/proverka-gipotez/kriterij-soglasiya-pirsona-khi-kvadrat/
# ----------------------------------------------------------------------------------------------------------------------
import numpy
from scipy.stats import chi2, chisquare
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
def ex_significance_level():
    a_significance_level = 0.05
    deg_of_freedom = 3
    critical_value = chi2.ppf(1-a_significance_level, deg_of_freedom)
    a_significance_level2 = 1-chi2.cdf(critical_value, deg_of_freedom)
    print(critical_value)
    print(a_significance_level2)
    return
# ----------------------------------------------------------------------------------------------------------------------
def plot_chi2_stats(stat_value_chi2,deg_of_freedom,a_significance_level=0.05):

    critical_value = chi2.ppf(1 - a_significance_level, deg_of_freedom)
    X_good = numpy.linspace(0, critical_value, 50)
    X_bad = numpy.linspace(critical_value, critical_value * 2, 50)

    Y_good = chi2.pdf(X_good, deg_of_freedom)
    Y_bad = chi2.pdf(X_bad, deg_of_freedom)

    plt.plot(X_good, Y_good, color=(0, 0, 0.5))
    plt.plot(X_bad, Y_bad, color=(0.5, 0, 0))

    plt.fill_between(X_good, Y_good, color=(0.7, 0.8, 1))
    plt.fill_between(X_bad, Y_bad, color=(1, 0.9, 0.9))

    marker = 'bo' if stat_value_chi2<critical_value else 'ro'

    plt.plot(stat_value_chi2,0,marker)
    plt.tight_layout()
    plt.show()
    return
# ----------------------------------------------------------------------------------------------------------------------
def is_fit_by_critical_value(X_obs, X_exp,a_significance_level=0.05,deg_of_freedom=None):

    if deg_of_freedom is None:
        deg_of_freedom = len(X_exp) - 1

    stat_value_chi2, p_value0  = chisquare(X_obs, X_exp)
    critical_value = chi2.ppf(1 - a_significance_level, deg_of_freedom)

    if stat_value_chi2< critical_value:
        print('stat_value_chi2 < critical_value')
        print('%1.2f            < %1.2f' % (stat_value_chi2, critical_value))
        print('Accept H0: fit OK')
    else:
        print("critical_value <= stat_value_chi2")
        print('%1.2f          <= %1.2f' % (critical_value, stat_value_chi2))
        print('Reject H0: no fit')

    print()
    return
# ----------------------------------------------------------------------------------------------------------------------
def is_fit_by_p_value(X_obs, X_exp,a_significance_level=0.05,deg_of_freedom=None):

    if deg_of_freedom is None:
        deg_of_freedom = len(X_exp) - 1

    stat_value_chi2, p_value0 = chisquare(X_obs, X_exp)
    p_value = chi2.sf(stat_value_chi2, deg_of_freedom)

    if p_value <= a_significance_level:
        print("p value <= significance_level")
        print('%1.2f    < %1.2f' % (p_value, a_significance_level))
        print('Reject H0: no fit')
    else:
        print("significance_level < p value")
        print('%1.2f               < %1.2f' % (a_significance_level, p_value))
        print('Accept H0: fit OK')
    print()

    return
# ----------------------------------------------------------------------------------------------------------------------
def analyze_experiment(X_obs,X_exp):

    deg_of_freedom = len(X_exp) - 1
    a_significance_level = 0.05

    stat_value_chi2, p_value0 = chisquare(X_obs, X_exp)
    is_fit_by_critical_value(X_obs, X_exp,a_significance_level)
    is_fit_by_p_value(X_obs, X_exp,a_significance_level)
    plot_chi2_stats(stat_value_chi2,deg_of_freedom,a_significance_level)

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_dice():
    X_obs = [8, 12, 13, 7, 12, 18]
    X_exp = [10, 10, 10, 10, 10, 10]
    analyze_experiment(X_obs,X_exp)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_coin():
    X_obs = [110, 90]
    X_exp = [100, 100]
    analyze_experiment(X_obs,X_exp)
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':


    ex_coin()
    #ex_dice()
