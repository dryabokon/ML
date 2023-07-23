from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import log_loss,  confusion_matrix
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
def get_data(n_samples=2500, n_features = 13):

    X, Y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=5, n_redundant=5, n_classes=3,random_state=1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5)

    return X_train, Y_train, X_test, Y_test
# ----------------------------------------------------------------------------------------------------------------------
def ex_regression_out_of_box(X_train, Y_train, X_test, Y_test):

    regr = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    regr.fit(X_train, Y_train)

    Y_pred_train = regr.predict(X_train).flatten()
    Y_pred_test  = regr.predict(X_test).flatten()
    Y_pred_prob_train = regr.predict_proba(X_train)
    Y_pred_prob_test = regr.predict_proba(X_test)

    print('Method       \tTrain\tTest\n' + '-' * 30)
    print('Cross H Loss:\t%1.4f\t%1.4f'%(log_loss(Y_train, Y_pred_prob_train),log_loss(Y_test, Y_pred_prob_test)))

    print('confusion_matrix train')
    print(confusion_matrix(Y_train, Y_pred_train))
    print('confusion_matrix test')
    print(confusion_matrix(Y_test, Y_pred_test))

    print()
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    X_train, Y_train, X_test, Y_test = get_data()
    ex_regression_out_of_box(X_train, Y_train, X_test, Y_test)