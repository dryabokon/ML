import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes,load_wine
from xgboost import XGBClassifier
from xgboost import plot_importance
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    dataset = load_wine()
    X,Y = dataset.data, dataset.target
    model = XGBClassifier()
    model.fit(X, Y)

    feature_importances = model.get_booster().get_score()

    print(feature_importances)
    plot_importance(model)
    plt.show()