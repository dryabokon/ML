import numpy
import pandas as pd
from scipy import stats
# ----------------------------------------------------------------------------------------------------------------------
import tools_DF
import tools_Hyptest
HypTest = tools_Hyptest.HypTest()
# ----------------------------------------------------------------------------------------------------------------------
def ex_2_dist():
    M1 = [10,10,10,10,10]
    M2 = [ 1, 5,9 ,10,11]

    sigma = 2

    res = []
    for m1,m2 in zip(M1,M2):
        S1 = pd.Series(numpy.random.normal(m1, sigma, 1000))
        S2 = pd.Series(numpy.random.normal(m2, sigma, 10))
        print(S2.values)
        res1 = stats.ks_2samp(S1.values, S2.values).pvalue
        res2 = HypTest.f1_score(S1, S2, is_categorical=False)
        res.append([m1,m2,res1,res2])

    df= pd.DataFrame(res,columns=['M1','M2','p_value(is same dist)','F1'])
    print(tools_DF.prettify(df,showindex=False))
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    m1= 10
    sigma = 2
    S1 = pd.Series(numpy.random.normal(m1, sigma, 1000))
    result, p_value = HypTest.is_same_distribution_chi2([5], S1.values)