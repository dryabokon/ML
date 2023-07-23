import cv2
import os
from random import random, randint
import pandas as pd

import numpy
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
# ---------------------------------------------------------------------------------------------------------------------
import mlflow
from mlflow import log_metric, log_param, log_artifacts,log_artifact, log_figure,log_image,log_text,artifacts
from mlflow.tracking import MlflowClient
# ---------------------------------------------------------------------------------------------------------------------
folder_out = './mlruns'
# ---------------------------------------------------------------------------------------------------------------------
def get_experiment_id(experiment_name, create=True):
    experiement = mlflow.get_experiment_by_name(experiment_name)
    if experiement is not None:
        experiment_id = experiement.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experiment_name) if create else None

    return experiment_id
# ---------------------------------------------------------------------------------------------------------------------
def run_experiment_01_dummy(experiment_name,artifact_path=None):
    mlflow.end_run()

    #ts = pd.Series(pd.DatetimeIndex(pd.Timestamp.now()).strftime('%Y-%b-%d %H:%M:%S'))
    ts = pd.Timestamp.now().strftime('%Y-%b-%d %H:%M:%S')

    #client = mlflow.tracking.MlflowClient()

    with mlflow.start_run(experiment_id=get_experiment_id(experiment_name, create=True), run_name=ts) as run:
        print('exp_id:', run.info.experiment_id)
        print('run_id:',run.info.run_id)
        log_param("param1", randint(0, 100))
        log_metric("foo", random())
        log_metric("foo", random() + 1)
        log_metric("foo", random() + 2)
        local_path = './data/output/brg.png'
        cv2.imwrite(local_path, numpy.full((320, 240, 3), 255, dtype=numpy.uint8))
        log_artifact(local_path)
        log_artifact(local_path=local_path,artifact_path=artifact_path)
        # if artifact_path is not None:
        #     os.system('gsutil cp %s %s'%(local_path,artifact_path))

        mlflow.end_run()
    return
# ---------------------------------------------------------------------------------------------------------------------
def run_experiment_02_epochs(experiment_name):
    mlflow.end_run()
    with mlflow.start_run(experiment_id=get_experiment_id(experiment_name, create=True)):
        for epoch in range(0, 3):
            mlflow.log_metric(key="F1", value=random(), step=epoch)
        mlflow.end_run()
    return
# ---------------------------------------------------------------------------------------------------------------------


def run_experiment_03_sklearn_RF(experiment_name):
    mlflow.end_run()
    with mlflow.start_run(experiment_id=get_experiment_id(experiment_name, create=True)) as run:
        print('exp_id:', run.info.experiment_id)
        print('run_id:', run.info.run_id)
        mlflow.autolog()
        db = load_diabetes()
        X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)
        rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
        rf.fit(X_train, y_train)
        rf.predict(X_test)
        image = numpy.full((320,240,3),255,dtype=numpy.uint8)
        cv2.imwrite('./data/output/histo_age.png',image)
        log_artifact(local_path='./data/output/histo_age.png')
        mlflow.last_active_run()
        mlflow.end_run()
    return
# ---------------------------------------------------------------------------------------------------------------------
def run_experiment_04_sklearn_DT_log_model(experiment_name):
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    experiment_id = get_experiment_id(experiment_name, create=True)
    mlflow.end_run()
    for idx, depth in enumerate([2,3]):
        with mlflow.start_run(experiment_id=experiment_id, run_name='depth_%d' % depth) as run:
            print('experm:', experiment_name)
            print('exp_id:', run.info.experiment_id)
            print('run_id:', run.info.run_id)
            clf = DecisionTreeClassifier(max_depth=depth)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_param("depth", depth)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(sk_model=clf,artifact_path="sklearn-model",registered_model_name="sk-learn-random-forest-reg-model")
            log_artifact(local_path='./data/output/brg.png')
            mlflow.end_run()

    return
# ---------------------------------------------------------------------------------------------------------------------
def ex06():
    print("Current tracking uri: %s"%mlflow.get_tracking_uri())
    # mlflow.set_tracking_uri("file:///tmp/my_tracking")
    return
# ---------------------------------------------------------------------------------------------------------------------
def set_tracking_local(folder_out):
    #Runs are recorded locally
    mlflow.set_tracking_uri(folder_out)
    mlflow.set_registry_uri(folder_out)
    return
# ---------------------------------------------------------------------------------------------------------------------
def set_tracking_remote(connection_string):
    # Runs are recorded remotely, Database is encoded as <dialect>+<driver>://<username>:<password>@<host>:<port>/<database>
    mlflow.set_tracking_uri(connection_string)
    return
# ---------------------------------------------------------------------------------------------------------------------
def get_uris():

    print('is_tracking_uri_set:', mlflow.tracking.is_tracking_uri_set())
    print('tracking.get_tracking_uri:', mlflow.tracking.get_tracking_uri())
    print('registry_uri:',mlflow.get_registry_uri())
    print('tracking_uri:',mlflow.get_tracking_uri())

    artifact_uri = mlflow.get_artifact_uri()
    print('artifact_uri:',artifact_uri)

    return artifact_uri
# ---------------------------------------------------------------------------------------------------------------------
def ex_tracking_local():
    set_tracking_local(folder_out)
    run_experiment_01_dummy(experiment_name='ex01')
    os.system('mlflow ui --backend-store-uri %s'%folder_out)

    return
# ---------------------------------------------------------------------------------------------------------------------
def ex_tracking_remote(connection_string):

    set_tracking_remote(connection_string)
    run_experiment_03_sklearn_RF(experiment_name='ex03_RF')

    # command = 'mlflow ui --backend-store-uri %s' % connection_string
    # print(command)
    #os.system()
    return
# ---------------------------------------------------------------------------------------------------------------------
#artifact_uri = mlflow.get_artifact_uri()
#mlflow.artifacts._download_artifact_from_uri('gs://testproj2-bf028.appspot.com/0/1a600a99c61a4bc985ac95b84e23acf1/artifacts/histo_alone.png', folder_out)
if __name__ == "__main__":

    # connection_string = 'http://34.122.102.32:5000'
    # set_tracking_remote(connection_string)
    set_tracking_local(folder_out)
    run_experiment_01_dummy(experiment_name='CI: integration tests')
    run_experiment_03_sklearn_RF(experiment_name='Featurestore')
    print('mlflow server --backend-store-uri %s --default-artifact-root %s'%(folder_out,folder_out))