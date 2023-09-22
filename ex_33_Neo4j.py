import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_neo4j
import tools_DF
import tools_plot_v2
from classifier import classifier_LM
import tools_ML_v2
# ----------------------------------------------------------------------------------------------------------------------
#filename_config_neo4j = './private_config_neo4j.yaml'
filename_config_neo4j = './public_config_neo4j.yaml'
filename_config_ssh = './private_config_ssh.yaml'
NEO4J = tools_neo4j.processor_Neo4j(filename_config_neo4j,filename_config_ssh,folder_out)
P = tools_plot_v2.Plotter(folder_out,dark_mode=False)
ML = tools_ML_v2.ML(classifier_LM.classifier_LM(),folder_out)
ML.P.set_color(0, P.color_blue)
ML.P.set_color(1, P.color_red)
# ----------------------------------------------------------------------------------------------------------------------
def prepare_dataset_random():
    X, Y = make_regression(n_samples=1250, n_features=3, noise=50.0)
    Y[Y <= 0] = 0
    Y[Y > 0] = 1
    df = pd.DataFrame(numpy.concatenate([X*100, Y.reshape((-1, 1))],axis=1),columns=['C%d' % c for c in range(X.shape[1])] + ['target']).astype(int)

    dct_entities = {'Item': [c for c in df.columns][:-1]}
    dct_relations = {'REL': ['Item', 'Item']}
    features_classification = [c for c in df.columns][:-1]
    target = 'target'

    return df, dct_entities, dct_relations, features_classification,target
# ----------------------------------------------------------------------------------------------------------------------
def prepare_dataset_titanic(flat_struct_for_classification=True):
    filename_in = folder_in + 'dataset_titanic.csv'
    df = pd.read_csv(filename_in, sep='\t')
    df.drop(columns=['alive', 'deck'], inplace=True)
    df['ID']=numpy.arange(df.shape[0])
    df = tools_DF.hash_categoricals(df)
    df = df.dropna()
    df = df.astype(int)
    target = 'survived'

    if flat_struct_for_classification:
        dct_entities = {'Person': ['ID', 'age','sex', 'who', 'adult_male','embark_town', 'embarked','parch', 'alone','sibsp','pclass', 'class','fare',target]}
        dct_relations = {'ORG': ['Person', 'Person']}
    else:
        dct_entities = {'Person': ['ID','age',target],
                        'Gender': ['sex', 'who', 'adult_male'],
                        'Location': ['embark_town', 'embarked'],
                        'Family': ['parch', 'alone','sibsp'],
                        'Cabin':['pclass', 'class','fare']
                        }
        dct_relations = {'ORG': ['Person', 'Location'],
                         'GND': ['Person', 'Gender'],
                         'FAM': ['Person', 'Family'],
                         'CAB': ['Person', 'Cabin'],
                         }

    features_classification = [f for f in dct_entities[[k for k in dct_entities.keys()][0]] if f!=target]

    return df,dct_entities,dct_relations,features_classification,target
# ----------------------------------------------------------------------------------------------------------------------
def prepare_dataset_heart():
    df, idx_target = pd.read_csv(folder_in + 'dataset_heart.csv', sep=','), -1
    dct_entities = {'Person':['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']}
    dct_relations = {'KNOWS': ['Person', 'Person']}
    features_classification = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
    target = 'target'
    return df,dct_entities,dct_relations,features_classification,target
# ----------------------------------------------------------------------------------------------------------------------
def prepare_dataset_trips(flat_struct_for_classification=True):
    filename_in = folder_in + 'df_trips_weekly.csv'
    df = pd.read_csv(filename_in)
    df['anomaly_level'].apply(lambda x: 1 if x>0 else 0).astype(int)
    numeric_series = pd.to_numeric(df['location'], errors='coerce')
    df = df[~numeric_series.isna()].astype(int)
    target = 'anomaly_level'

    if flat_struct_for_classification:
        dct_entities = {'Person': ['username','connections', 'connections_valid', 'is_valid_roundtrip', 'are_intervals_valid', 'mileage_driven','miles_coords', 'duration', 'max_remote',target],
                        'Location': ['location']}
        dct_relations = {'ORG': ['Person', 'Location']}
    else:
        dct_entities = {
            'Person'     : ['username'],
            'Location'   : ['location'],
            'Connections': ['connections', 'connections_valid'],
            'Mileage'    : ['mileage_driven','miles_coords','max_remote'],
            'Duration'   : ['duration'],
            'Aligned'    : [ 'is_valid_roundtrip', 'are_intervals_valid']
        }
        dct_relations = {'ORG': ['Person', 'Location'],
                         'CON': ['Person', 'Connections'],
                         'MIL': ['Person', 'Mileage'],
                         'TIM': ['Person', 'Duration'],
                         'ALG': ['Person', 'Aligned'],
                         }

    features_classification = [f for f in dct_entities[[k for k in dct_entities.keys()][0]] if f != target]
    return df,dct_entities,dct_relations,features_classification,target
# ----------------------------------------------------------------------------------------------------------------------
def ex_embeddings_native(df,dct_entities, dct_relations,target,emb_dims = 64):
    entity = [k for k in dct_entities.keys()][0]
    identity = [v for v in dct_entities[entity]][0]

    NEO4J.export_df_to_neo4j(df,dct_entities, dct_relations,drop_if_exists=True)
    graph_name = NEO4J.create_graph_native(dct_entities,dct_relations)
    NEO4J.run_fastRP_encoder_native(graph_name=graph_name, entity=entity, identity=identity, emb_dims=emb_dims)
    # model_name = NEO4J.train_graphSage_encoder_native(graph_name,dct_entities, dct_relations, features, emb_dims)
    # NEO4J.run_graphSage_encoder_native(graph_name, model_name, entity, identity, dct_entities, dct_relations)

    df_E = NEO4J.fetch_output_from_neo4j(identity=identity, filename_out='df_E.csv')
    df_Y = tools_DF.fetch(df, identity, df_E, identity, col_value=[c for c in df_E.columns][1:])
    #df_Y.to_csv(folder_out + 'df_Y.csv',index = False)
    #df_Y = pd.read_csv(folder_out + 'df_Y.csv')
    df_ET = pd.concat([df_Y[target], df_Y.iloc[:, -emb_dims:]], axis=1)
    P.plot_PCA(df_ET, idx_target=df_ET.columns.get_loc(target), method='tSNE', filename_out='Embeddings_tSNE.png')

    #P.plot_2D_features(df_Y.iloc[:,[df_Y.columns.get_loc(target)]+[-(i) for i in range(1,1+emb_dims)]], add_noice=True,filename_out='Embeddings_Neo4j.png')
    #P.plot_PCA(df, idx_target=df.columns.get_loc(target),method='tSNE', filename_out='Embeddings_tSNE.png')
    # P.plot_PCA(df, idx_target=df.columns.get_loc(target),method='UMAP', filename_out='Embeddings_UMAP.png')
    NEO4J.close()
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_classification_gds(df_train,df_test,dct_entities, dct_relations,features):
    entity = [k for k in dct_entities.keys()][0]
    identity = [v for v in dct_entities[entity]][0]
    target_property = [v for v in dct_entities[entity]][-1]

    NEO4J.export_df_to_neo4j(df_train,dct_entities, dct_relations,drop_if_exists=True)
    G_train = NEO4J.create_graph_gds(dct_entities, dct_relations)
    model = NEO4J.train_model_classification_gds(G_train, entity=entity, target_property=target_property, features=features)

    NEO4J.export_df_to_neo4j(df_test,dct_entities, dct_relations,drop_if_exists=True)
    G_test = NEO4J.create_graph_gds(dct_entities, dct_relations)
    NEO4J.run_model_classification_gds(model, G_test, entity=entity, identity=identity)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_classification_naive(df_train,df_test,dct_entities, dct_relations,features,target):
    entity = [k for k in dct_entities.keys()][0]
    identity = [v for v in dct_entities[entity]][0]

    idx_target = df.columns.get_loc(target)
    NEO4J.export_df_to_neo4j(df_train,dct_entities, dct_relations,drop_if_exists=True)
    graph_train = NEO4J.create_graph_native(dct_entities,dct_relations)
    model_name = NEO4J.train_model_classification_native(graph_train,entity=entity, target_property=target,features=features)
    NEO4J.run_model_classification_native(graph_name=graph_train, model_name=model_name, entity=entity,identity=identity)
    scores_train = 100*NEO4J.fetch_output_from_neo4j(identity=identity, filename_out='df_train.csv').iloc[:, -1].values
    dct_metrics_train = ML.evaluate_metrics(df_train, idx_target, scores=scores_train, is_train=True, plot_charts=True)

    NEO4J.export_df_to_neo4j(df_test,dct_entities, dct_relations,drop_if_exists=True)
    graph_test = NEO4J.create_graph_native(dct_entities, dct_relations)
    NEO4J.run_model_classification_native(graph_name=graph_test,model_name=model_name,entity=entity,identity=identity)
    scores_test = 100*NEO4J.fetch_output_from_neo4j(identity=identity, filename_out='df_test.csv').iloc[:,-1].values
    dct_metrics_test  = ML.evaluate_metrics(df_test, idx_target, scores=scores_test, is_train=False,plot_charts=True)

    df_metrics = ML.combine_metrics(dct_metrics_train, dct_metrics_test,None)
    df_metrics = pd.concat([pd.DataFrame(numpy.array(['#', df_train.shape[0], df_test.shape[0]]).reshape((1, -1)),columns=df_metrics.columns), df_metrics])
    print(tools_DF.prettify(df_metrics, showindex=False))
    NEO4J.close()
    return
# ----------------------------------------------------------------------------------------------------------------------
#df,dct_entities, dct_relations,features,target  = prepare_dataset_titanic(flat_struct_for_classification=False)
#df,dct_entities, dct_relations,features,target = prepare_dataset_random()
#df,dct_entities, dct_relations, features,target = prepare_dataset_heart()
df,dct_entities, dct_relations, features,target = prepare_dataset_trips(flat_struct_for_classification=False)
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    #tools_IO.remove_files(folder_out)
    df_train, df_test = train_test_split(df, test_size=0.5, shuffle=True)
    ex_embeddings_native(df, dct_entities, dct_relations,target,emb_dims = 16)
