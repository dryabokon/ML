import tools_Langchain
# ----------------------------------------------------------------------------------------------------------------------
# filename_config_chat='./secrets/private_config_openai.yaml'
# filename_config_emb='./secrets/private_config_openai.yaml'
# ----------------------------------------------------------------------------------------------------------------------
filename_config_chat='./secrets/private_config_azure_chat.yaml'
filename_config_emb='./secrets/private_config_azure_embeddings.yaml'
# ----------------------------------------------------------------------------------------------------------------------
filename_config_vectorstore = './secrets/private_config_azure_search.yaml'
# ----------------------------------------------------------------------------------------------------------------------
A = tools_Langchain.Assistant(filename_config_chat_model=filename_config_chat, filename_config_emb_model=filename_config_emb,filename_config_vectorstore=filename_config_vectorstore,chain_type='Summary')
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    A.add_document('./data/ex_LLM/Godfather.txt',azure_search_index_name='idx-godfather')
    res = A.run_chain('What is MIke\'s hobby?', azure_search_index_name='idx-godfather',do_debug=True)
    print(res)
