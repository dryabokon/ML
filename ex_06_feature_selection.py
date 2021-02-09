import numpy
import seaborn as sns
from sklearn.feature_selection import chi2
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    df = sns.load_dataset('titanic')
    df = df.dropna()
    df = df[['survived','age', 'pclass', 'sex', 'sibsp', 'parch', 'embarked', 'who', 'alone']]
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    df['who'] = df['who'].map({'man': 0, 'woman': 1, 'child': 2})
    df['alone'] = df['alone'].map({True: 1, False: 0})

    X_df = df[['age', 'pclass', 'sex', 'sibsp', 'parch', 'embarked', 'who', 'alone']]
    Y = df[['survived']].to_numpy()

    f_scores = chi2(X_df.to_numpy(), Y)[1]
    idx =  numpy.argsort(-f_scores)

    print('corr\tf_score\tfeature_name')
    for feature_name,f_score in zip(X_df.columns.to_numpy()[idx], f_scores[idx]):
        cc = numpy.abs(numpy.corrcoef(Y.flatten(), X_df[[feature_name]].to_numpy().flatten())[0,1])
        print('%1.2f\t%1.2f\t%s'%(cc,f_score,feature_name))
