import numpy
import pandas as pd
import requests
from elasticsearch import Elasticsearch, helpers
from sklearn import metrics
# ----------------------------------------------------------------------------------------------------------------------
import tools_ML_v2
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
def get_models():
    res = requests.get('http://34.121.12.66:9200/_ml/trained_models/_all/')
    print(res.json())
    return
# ----------------------------------------------------------------------------------------------------------------------
def get_records(host_port,index_name,fields=None,limit=100):

    es = Elasticsearch(host_port)
    query = {"query": {"match_all": {}}}
    if fields is not None:query['_source'] = fields

    l,dct_res = 0,[]
    for rec in helpers.scan(es,query=query,index=index_name,ignore_unavailable=True):
        dct_res.append(dict(rec['_source']))
        l+=1
        if l>limit:break

    df = pd.json_normalize(dct_res)
    return df
# ----------------------------------------------------------------------------------------------------------------------
def evaluate_accuracy(Y, scores, is_train):
    ML = tools_ML_v2.ML(None, folder_out)

    fpr, tpr, thresholds = metrics.roc_curve(Y, scores)
    auc = metrics.auc(fpr, tpr)
    accuracy = ML.get_accuracy(tpr, fpr, nPos=numpy.count_nonzero(Y > 0), nNeg=numpy.count_nonzero(Y <= 0))

    if is_train:
        caption = 'Train'
    else:
        caption = 'Test'

    ML.P.plot_tp_fp(tpr, fpr, auc, caption=caption, filename_out=caption + '_auc.png')
    print('ACC_%s = %1.3f' % (caption, accuracy))
    print('AUC_%s = %1.3f' % (caption, auc))
    return
# ----------------------------------------------------------------------------------------------------------------------
host_port='http://34.121.12.66:9200'
# ----------------------------------------------------------------------------------------------------------------------
# index_name='flightclassification'
# fields=['Cancelled','ml.prediction_probability','ml.is_training']
# ----------------------------------------------------------------------------------------------------------------------
index_name='kibana_sample_data_logs'
fields=None
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    df = get_records(host_port,index_name,fields,limit=100000)
    df = df.replace({'\t': ' '})
    df.to_csv(folder_out + 'df_pred.txt', index=False, sep='\t')

    # Y = 1*(df.iloc[:,0].to_numpy())
    # scores = 1-df.iloc[:,1].to_numpy()
    # is_train = df.iloc[:,2].to_numpy()
    #
    # evaluate_accuracy(Y[ is_train], scores[ is_train],True)
    # evaluate_accuracy(Y[~is_train], scores[~is_train],False)





