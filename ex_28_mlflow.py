import os
from random import random, randint
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
# ---------------------------------------------------------------------------------------------------------------------
import mlflow
from mlflow import log_metric, log_param, log_artifacts, log_figure,log_image,log_text
# ---------------------------------------------------------------------------------------------------------------------
def ex01():
    log_param("param1", randint(0, 100))
    log_metric("foo", random())
    log_metric("foo", random() + 1)
    log_metric("foo", random() + 2)
    log_artifacts('./data/output2d/')
    return
# ---------------------------------------------------------------------------------------------------------------------
def ex02():
    active_run = mlflow.start_run()
    for epoch in range(0, 3):
        mlflow.log_metric(key="F1", value=random(), step=epoch)
    return
# ---------------------------------------------------------------------------------------------------------------------
def ex03():
    mlflow.autolog()
    db = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

    rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    autolog_run = mlflow.last_active_run()
    return
# ---------------------------------------------------------------------------------------------------------------------
def ex04():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    EXPERIMENT_NAME = "mlflow-demo"
    EXPERIMENT_ID = '1'
    #EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)

    for idx, depth in enumerate([1, 2, 5, 10, 20]):
        clf = DecisionTreeClassifier(max_depth=depth)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name='run_%d'%idx) as run
            RUN_ID = run.info.run_id
            mlflow.log_param("depth", depth)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(clf, "classifier")



    return
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    ex04()





