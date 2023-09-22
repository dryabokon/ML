import tools_Langchain
# ----------------------------------------------------------------------------------------------------------------------
#filename_in = './data/ex_LLM/MBA_CM/01-how-to-deal-with-resistance-to-change.txt'
# ----------------------------------------------------------------------------------------------------------------------
# filename_config_chat='./secrets/private_config_openai.yaml'
# filename_config_emb='./secrets/private_config_openai.yaml'
# ----------------------------------------------------------------------------------------------------------------------
filename_config_chat='./secrets/private_config_azure_chat.yaml'
filename_config_emb='./secrets/private_config_azure_embeddings.yaml'
# ----------------------------------------------------------------------------------------------------------------------
#filename_config_vectorstore = './secrets/private_config_pinecone.yaml'
filename_config_vectorstore = './secrets/private_config_azure_search.yaml'
# ----------------------------------------------------------------------------------------------------------------------
A = tools_Langchain.Assistant(filename_config_chat_model=filename_config_chat, filename_config_emb_model=filename_config_emb,filename_config_vectorstore=filename_config_vectorstore,chain_type='Summary')
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    A.add_document_azure('./data/ex_LLM/prompt1.txt')

    # for filename in ['01-how-to-deal-with-resistance-to-change.txt','02-forbes.txt','03_LI.txt','04_walkme.txt','05_voltagecontrol.txt','06_udemy.txt','07_prosci.txt','08_niagarainstitute.txt']:
    #     A.add_document_to_pinecone_index('./data/ex_LLM/MBA_CM/'+filename,text_key='CM')

    # res = A.run_query('Summarize what you have learned about managing it and facilitating the changes' ,text_key='CM')
    # res = A.run_query('Provide detailed summary about the Sales strategies' ,text_key='MBA')

    # text_key = 'Godfather_Azure'
    #res = A.run_chain('where do homeless people stay??', text_key='Hyperawareness',do_debug=True)
    #print(res)
