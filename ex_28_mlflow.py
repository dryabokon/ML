import cv2
import os
import subprocess
import sys
import yaml
from random import random, randint
import pandas as pd

if sys.platform == 'win32':
    # mlflow prints an emoji in its run-URL banner, which crashes on the default cp1251 console encoding
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

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
from mlflow.entities import SpanType
# ---------------------------------------------------------------------------------------------------------------------
folder_out = './mlruns'
filename_config_mlflow = './secrets/private_config_mlflow.yaml'
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
def run_experiment_05_llm_usage(experiment_name):
    # Simulates LLM calls (no real API key needed) so the experiment's "Usage" tab in the MLflow UI,
    # which aggregates token counts from GenAI traces, has data to show.
    # NB: mlflow.start_span(..., trace_destination=...) is broken in mlflow==3.5.1 (it accesses the
    # non-existent TraceLocation.experiment_id and silently falls back to a no-op span), so the
    # active experiment is set explicitly instead and the default destination is used.
    experiment_id = get_experiment_id(experiment_name, create=True)
    mlflow.set_experiment(experiment_id=experiment_id)

    prompts_and_answers = [
        ("Summarize MLflow in one sentence.", "MLflow tracks experiments, models, and artifacts for ML projects."),
        ("What does the Usage tab show?", "It shows token counts aggregated from GenAI traces."),
    ]

    for prompt, answer in prompts_and_answers:
        input_tokens, output_tokens = len(prompt.split()), len(answer.split())
        with mlflow.start_span(name="chat_completion", span_type=SpanType.LLM) as span:
            span.set_inputs({"messages": [{"role": "user", "content": prompt}]})
            span.set_outputs({"choices": [{"message": {"role": "assistant", "content": answer}}]})
            span.set_attribute("mlflow.chat.tokenUsage", {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens})
            print('trace_id:', span.trace_id)

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
def set_tracking_remote_auth(filename_config):
    # Runs are recorded on a remote MLflow tracking server protected with HTTP basic auth.
    # Credentials are kept out of source control in a local yaml config (see secrets/private_config_mlflow.yaml).
    with open(filename_config, 'r') as f:
        cfg = yaml.safe_load(f)['mlflow']

    os.environ['MLFLOW_TRACKING_USERNAME'] = cfg['username']
    os.environ['MLFLOW_TRACKING_PASSWORD'] = cfg['password']
    mlflow.set_tracking_uri(cfg['host'])

    return cfg['host']
# ---------------------------------------------------------------------------------------------------------------------
def scp_artifact_to_remote_mlflow(filename_config, experiment_id, run_id, local_path, remote_filename=None):
    # Bypasses the MLflow HTTP artifact proxy and writes straight onto the tracking server's disk over
    # scp/ssh (via gcloud, since the VM uses OS Login rather than a static keypair). This is the path to
    # reach for large files (model weights, datasets) where per-file HTTP upload overhead adds up -
    # the server's LocalArtifactRepository lists whatever it finds on disk under <run>/artifacts, so
    # nothing needs to be registered separately once the file is in place.
    with open(filename_config, 'r') as f:
        cfg = yaml.safe_load(f)['mlflow']

    remote_filename = remote_filename or os.path.basename(local_path)
    remote_dir = '%s/%s/%s/artifacts' % (cfg['remote_artifact_root'], experiment_id, run_id)
    staging_path = '/tmp/%s' % remote_filename

    subprocess.run(['gcloud', 'compute', 'scp', local_path, '%s:%s' % (cfg['gcloud_instance'], staging_path),
                     '--zone', cfg['gcloud_zone']], check=True)

    # the artifact store's directories are created root-owned by the mlflow server container, so a plain
    # scp into them would fail with permission denied - stage to /tmp instead, then move into place with sudo.
    remote_cmd = 'sudo mkdir -p %s && sudo cp %s %s/%s && sudo chown root:root %s/%s && rm %s' % (
        remote_dir, staging_path, remote_dir, remote_filename, remote_dir, remote_filename, staging_path)
    subprocess.run(['gcloud', 'compute', 'ssh', cfg['gcloud_instance'], '--zone', cfg['gcloud_zone'],
                     '--command', remote_cmd], check=True)

    return '%s/%s' % (remote_dir, remote_filename)
# ---------------------------------------------------------------------------------------------------------------------
def run_experiment_06_scp_large_artifact(experiment_name, filename_config):
    # Showcases the scp-based transfer path: params/metrics still go through the normal REST API (they're
    # tiny), but the artifact is placed directly onto the tracking server's disk via scp+ssh instead of
    # mlflow.log_artifact()'s HTTP upload. Swap local_path for an actual model checkpoint/dataset to use
    # this for real - the demo file here is a stand-in so the example runs without extra assets.
    mlflow.end_run()
    with mlflow.start_run(experiment_id=get_experiment_id(experiment_name, create=True), run_name='scp_demo') as run:
        print('exp_id:', run.info.experiment_id)
        print('run_id:', run.info.run_id)
        mlflow.log_param("transfer_method", "scp")

        local_path = './data/output/scp_demo_artifact.png'
        cv2.imwrite(local_path, numpy.full((64, 64, 3), 128, dtype=numpy.uint8))
        remote_path = scp_artifact_to_remote_mlflow(filename_config, run.info.experiment_id, run.info.run_id, local_path)
        print('scp-ed artifact to:', remote_path)

        mlflow.end_run()
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
def ex_tracking_remote_auth():
    # Showcases logging params/metrics/artifacts and a registered model against a hosted,
    # login-protected MLflow tracking server (see secrets/private_config_mlflow.yaml for credentials).
    host = set_tracking_remote_auth(filename_config_mlflow)

    run_experiment_01_dummy(experiment_name='CI: integration tests')
    run_experiment_03_sklearn_RF(experiment_name='Featurestore')
    run_experiment_04_sklearn_DT_log_model(experiment_name='ex04_DT_log_model')
    run_experiment_05_llm_usage(experiment_name='ex04_DT_log_model')
    run_experiment_06_scp_large_artifact(experiment_name='CI: integration tests', filename_config=filename_config_mlflow)

    run = mlflow.last_active_run()
    print('MLflow UI: %s/#/experiments/%s/runs/%s' % (host, run.info.experiment_id, run.info.run_id))
    print('Usage tab: %s/#/experiments/%s/overview/usage' % (host, get_experiment_id('ex04_DT_log_model', create=False)))

    return
# ---------------------------------------------------------------------------------------------------------------------
#artifact_uri = mlflow.get_artifact_uri()
#mlflow.artifacts._download_artifact_from_uri('gs://testproj2-bf028.appspot.com/0/1a600a99c61a4bc985ac95b84e23acf1/artifacts/histo_alone.png', folder_out)
if __name__ == "__main__":

    # set_tracking_local(folder_out)
    # run_experiment_01_dummy(experiment_name='CI: integration tests')
    # run_experiment_03_sklearn_RF(experiment_name='Featurestore')
    # print('mlflow server --backend-store-uri %s --default-artifact-root %s'%(folder_out,folder_out))

    ex_tracking_remote_auth()