#https://neo4j.com/developer-blog/harness-large-language-models-neo4j/
#https://neo4j.com/developer-blog/langchain-cypher-search-tips-tricks/
# ----------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
import tools_DF
import tools_Langchain
import tools_neo4j
# ----------------------------------------------------------------------------------------------------------------------
filename_config_neo4j = './private_config_neo4j.yaml'
filename_config_ssh = './private_config_ssh.yaml'
NEO4J = tools_neo4j.processor_Neo4j(filename_config_neo4j,filename_config_ssh,folder_out)
A = tools_Langchain.Assistant(pinecone_api_key='788a0515-aa52-4d1d-ac1c-a1ae44fc9460', pinecone_index_name="idx5", filename_openai_key='openaiapikey_private_D.txt', chain_type='Neo4j')
# ----------------------------------------------------------------------------------------------------------------------
def prepare_dataset_titanic(flat_struct_for_classification=True):
    filename_in = folder_in + 'dataset_titanic.csv'
    df = pd.read_csv(filename_in, sep='\t')
    df.drop(columns=['alive', 'deck'], inplace=True)
    df['ID']=numpy.arange(df.shape[0])
    #df = tools_DF.hash_categoricals(df)
    df = df.dropna()
    #df = df.astype(int)
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
def export_to_neo4j():
    df, dct_entities, dct_relations, features, target = prepare_dataset_titanic(flat_struct_for_classification=False)
    NEO4J.export_df_to_neo4j(df, dct_entities, dct_relations, drop_if_exists=True)
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    #export_to_neo4j()

    query = "How many survived males aged 50+ from Southampton?"
    #query = "how many woman aged 50+ from Southampton has survived"
    res = A.chain.run(query)
    print(query,res)
    NEO4J.close()







