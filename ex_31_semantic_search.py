import pandas as pd
import inspect
# ----------------------------------------------------------------------------------------------------------------------
import utils_semantic
# ----------------------------------------------------------------------------------------------------------------------
import sys
sys.path.insert(1, './tools/')
# ----------------------------------------------------------------------------------------------------------------------
import tools_time_profiler
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
TP = tools_time_profiler.Time_Profiler()
# ----------------------------------------------------------------------------------------------------------------------
def pipe_01_tokenize_words():
    TP.tic(inspect.currentframe().f_code.co_name, reset=True)
    U = utils_semantic.Semantic_proc(folder_out=folder_out)
    U.tokenize_words('./data/ex_words/labels_yolo.txt')
    TP.print_duration(inspect.currentframe().f_code.co_name)
    return
# ----------------------------------------------------------------------------------------------------------------------
def pipe_02a_tokenize_images():
    TP.tic(inspect.currentframe().f_code.co_name, reset=True)
    U = utils_semantic.Semantic_proc(folder_out=folder_out)
    U.tokenize_images('./data/ex_natural_images/airplane/')
    TP.print_duration(inspect.currentframe().f_code.co_name)
    return
# ----------------------------------------------------------------------------------------------------------------------
def pipe_02b_tokenize_URLs_images(limit =10000):
    TP.tic(inspect.currentframe().f_code.co_name, reset=True)
    U = utils_semantic.Semantic_proc(folder_out=folder_out,cold_start=True)
    df = pd.read_csv('./data/ex_GCC/Validation.tsv',sep='\t',header=None)
    captions = df.iloc[:limit, 0]
    URLs = df.iloc[:limit,1]

    U.tokenize_URLs_images(URLs,captions)
    TP.print_duration(inspect.currentframe().f_code.co_name)
    return
# ----------------------------------------------------------------------------------------------------------------------
def pipe_02c_tokenize_youtube(queries,limit=3):

    U = utils_semantic.Semantic_proc(folder_out=folder_out, cold_start=True)
    URLs = U.get_ULRs(queries,limit)
    #!!!
    URLs[0] = 'https://www.youtube.com/watch?v=g-skPkW75mQ&t=6s'
    U.tokenize_youtube_scenes(URLs,filename_log='df_tokens_youtube.csv')
    return
# ----------------------------------------------------------------------------------------------------------------------
def pipe_03_image_text_similarity():
    TP.tic(inspect.currentframe().f_code.co_name, reset=True)
    U = utils_semantic.Semantic_proc(folder_out=folder_out, cold_start=False)
    U.tokens_similarity('./data/ex_tokens_hex/tokens_labels_yolo.csv','./data/ex_tokens_hex/tokens_airplane.csv')
    TP.print_duration(inspect.currentframe().f_code.co_name)
    return
# ----------------------------------------------------------------------------------------------------------------------
def pipe_04_PCA():
    import tools_plot_v2
    P = tools_plot_v2.Plotter(folder_out=folder_out, dark_mode=False)
    df = pd.DataFrame([])
    for t in ['airplain','car','cat','dog','flower','fruit','motorbike','person']:
        df1 = pd.read_csv('./data/ex_tokens/tokens_%s.csv'%t)
        df1.iloc[:,0] = t
        df = pd.concat([df,df1],axis=0)

    P.plot_PCA(df, idx_target=0, method='tSNE',filename_out='PCA.png')
    return
# ----------------------------------------------------------------------------------------------------------------------
def pipe_05_search_image(query_text,top_n=6):
    TP.tic(inspect.currentframe().f_code.co_name, reset=True)
    U = utils_semantic.Semantic_proc(folder_out=folder_out, cold_start=False)

    filename_images = U.search_images(query_text=query_text,
                                      filename_tokens_images='./data/ex_GCC/tokens.csv',
                                      filename_tokens_words ='./data/ex_tokens_hex/tokens_english_3000.csv',top_n=top_n)
    filename_out = 'thumbnails.jpg'
    U.compose_thumbnails('./data/ex_GCC/', filename_images,filename_out)
    TP.print_duration(inspect.currentframe().f_code.co_name)
    return

# ----------------------------------------------------------------------------------------------------------------------

#pipe_01_tokenize_words()
#pipe_02a_tokenize_images()
#pipe_02b_tokenize_URLs_images()
#pipe_03_image_text_similarity()
#pipe_05_search_image('car2',4)
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    #pipe_06('cnn news')
    #pipe_02a_tokenize_images()
    #pipe_02b_tokenize_URLs_images()
    #pipe_01_tokenize_words()
    pipe_05_search_image('science')