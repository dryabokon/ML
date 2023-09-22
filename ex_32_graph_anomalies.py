import numpy
import pandas as pd
import tools_graph_data
import tools_IO
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
G = tools_graph_data.Graph_Processor(folder_out)
# ----------------------------------------------------------------------------------------------------------------------
def ex_karate():
    data = G.get_data_karate()
    node_size = 800
    G.draw_graph(data,filename_out='karate_shell.png',layout='shell',node_size=node_size)
    G.draw_graph(data,filename_out='karate_shell_sorted.png',layout='shell_sorted',node_size=node_size)
    G.draw_graph(data,filename_out='karate_sprng.png',layout='sprng',node_size=node_size)
    G.draw_mat(data, 'karate_mat_sorted.png',layout='shell_sorted')
    G.draw_mat(data, 'karate_mat.png', layout='shell')
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_facebook():
    data = G.get_data_facebook('./data/ex_GAD/facebook_combined.txt')
    node_size = 100
    alpha = 0.01
    G.draw_graph(data, filename_out='FB_shell.png',layout='shell', node_size=node_size,alpha=alpha)
    G.draw_graph(data, filename_out='FB_shell_sorted.png',layout='shell_sorted', node_size=node_size, alpha=alpha)
    G.draw_graph(data, filename_out='FB_sprng.png',layout='sprng', node_size=node_size,alpha=0.3)
    G.draw_mat(data, 'FB_mat_sorted.png', layout='shell_sorted')
    G.draw_mat(data, 'FB_mat.png', layout='shell')
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_IMDB():

    # movies - actors
    data = G.feature_engineering('./data/ex_datasets/dataset_IMDB.csv', idx_entity1=1, idx_entity2=5, idx_features1=[6, 8, 9, 10, 11], idx_features2=[8, 9, 10, 11], do_split1=False, do_split2=True)
    #data = G.normalize_data('./data/ex_datasets/dataset_IMDB.csv', idx_entity1=1, idx_entity2=5, idx_features1=[], idx_features2=[], do_split1=False, do_split2=True)
    #data = G.normalize_data('./data/ex_datasets/dataset_IMDB.csv', idx_entity1=5, idx_entity2=1, idx_features1=[],idx_features2=[], do_split1=True, do_split2=False)

    # actors - directors
    #data = G.normalize_data('./data/ex_datasets/dataset_IMDB.csv', idx_entity1=5, idx_entity2=4, idx_features1=[6,8,9,10,11], idx_features2=[6,8,9,10,11], do_split1=True, do_split2=False)

    # directors - actors
    #data = G.normalize_data('./data/ex_datasets/dataset_IMDB.csv', idx_entity1=4, idx_entity2=5,idx_features1=[6, 8, 9, 10, 11], idx_features2=[6, 8, 9, 10, 11], do_split1=False,do_split2=True)

    #G.draw_graph(data, filename_out=data.x.columns[1] + '_shell.png',layout='shell', node_size=100,alpha=0.1)
    #G.draw_graph(data, filename_out=data.x.columns[1] + '_sprng.png',layout='sprng', node_size=100,alpha=0.3)
    # G.draw_graph(data, filename_out=data.x.columns[1] + '_shell_sorted.png', layout='shell_sorted', node_size=100, alpha=0.1)
    # G.draw_mat(data, data.x.columns[1] + '.png', layout='shell')
    # G.draw_mat(data, data.x.columns[1]+'_sorted.png',layout='shell_sorted')

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_delivery():

    #short trip, long trip


    df = pd.read_csv('./data/ex_datasets/dataset_delivery2.csv')
    #dct = {'BookingID':0,'vehicle_no':2,'Origin_Location':3,'Destination_Location':4,'Driver_Name':11,'customerNameCode':13,'supplierNameCode':14}
    dct = {'vehicle_no': 2, 'Origin_Location': 3, 'Destination_Location': 4, 'Driver_Name': 11,'customerNameCode': 13, 'supplierNameCode': 14}
    keys = [k for k in dct.keys()]
    k1,k2= 3,0
    data = G.feature_engineering(df, idx_entity1=dct[keys[k1]], idx_entity2=dct[keys[k2]], idx_features1=[9], idx_features2=[9])
    G.draw_mat(data, filename_out='%s_%s_matr.png'%(keys[k1],keys[k2]), layout='shell_sorted')

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    tools_IO.remove_files(folder_out,'*.csv,*.png')
