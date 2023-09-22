import json
# ----------------------------------------------------------------------------------------------------------------------
import tools_neo4j
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
filename_config_neo4j = './private_config_neo4j.yaml'
filename_config_ssh = './private_config_ssh.yaml'
NEO4J = tools_neo4j.processor_Neo4j(filename_config_neo4j,filename_config_ssh,folder_out)
# ----------------------------------------------------------------------------------------------------------------------
def export_data_to_neo4j():


    CORA_CONTENT = "https://data.neo4j.com/cora/cora.content"
    CORA_CITES = "https://data.neo4j.com/cora/cora.cites"
    SUBJECT_TO_ID = {"Neural_Networks": 100, "Rule_Learning": 1, "Reinforcement_Learning": 2,
                     "Probabilistic_Methods": 3, "Theory": 4, "Genetic_Algorithms": 5, "Case_Based": 6}
    HOLDOUT_NODES = 10
    subject_map = json.dumps(SUBJECT_TO_ID).replace('"', "`")

    load_nodes = f"""
            LOAD CSV FROM "{CORA_CONTENT}" AS row
            WITH 
              {subject_map} AS subject_to_id,
              toInteger(row[0]) AS extId, 
              row[1] AS subject, 
              toIntegerList(row[2..]) AS features
            MERGE (p:Paper {{extId: extId, subject: subject_to_id[subject], features: features}})
            WITH p LIMIT {HOLDOUT_NODES}
            REMOVE p:Paper
            SET p:UnclassifiedPaper
        """

    load_relationships = f"""
            LOAD CSV FROM "{CORA_CITES}" AS row
            MATCH (n), (m) 
            WHERE n.extId = toInteger(row[0]) AND m.extId = toInteger(row[1])
            MERGE (n)-[:CITES]->(m)
        """

    NEO4J.execute_query('MATCH (n) DETACH DELETE n')
    NEO4J.gds.run_cypher(load_nodes)
    NEO4J.gds.run_cypher(load_relationships)
    return
# ----------------------------------------------------------------------------------------------------------------------
dct_entities = {'Paper': ['features', 'subject'],'UnclassifiedPaper': ['features']}
dct_relations = {'CITES': ['Paper', 'Paper']}
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #export_data_to_neo4j()
    G = NEO4J.create_graph(dct_entities, dct_relations)
    model = NEO4J.train_model(G,features="features",target_entity='Paper',target_property="subject")
    NEO4J.run_model(model, G, 'Paper', 'subject')

