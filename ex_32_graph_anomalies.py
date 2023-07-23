import numpy
import cv2
import tools_graph_data
import tools_animation
import tools_IO
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
G = tools_graph_data.Graph_Processor(folder_out)
# ----------------------------------------------------------------------------------------------------------------------
def ex_YelpRes():
    data = G.load_data_mat('./data/ex_GAD/YelpRes.mat')
    G.export_data_pandas(data,'YelpRes.csv')
    G.save_graph_pyvis(data,'YelpRes.html')
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_pygod():
    #data = G.get_data_planetoid()
    dataset = 'disney'
    data = G.get_data_pygod(dataset)
    G.export_data_pandas(data,dataset+'.csv')
    G.save_graph_pyvis(data,dataset+'.html')
    G.save_graph_gephi(data,dataset+'.gexf')
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_MUTAG():
    data = G.get_data_TUDataset()
    G.export_data_pandas(data,'MUTAG.csv')
    #G.save_graph_pyvis(data,'MUTAG.html')
    G.save_graph_gephi(data,'MUTAG.gexf')
    return
# ----------------------------------------------------------------------------------------------------------------------
#G.construct_animation(G.get_data_karate())
tools_animation.folder_to_animated_gif_imageio(folder_out, folder_out+'part1.gif', mask='*.png,*.jpg', framerate=10,stop_ms=4000,do_reverce=True)
# im = cv2.imread(folder_out+'FB_mat_sorted.png')
# for x in numpy.linspace(0,im.shape[0]-1000,100).astype(int):
#     cv2.imwrite(folder_out+'cut_%04d.png'%x,im[x:x+1000,x:x+1000])

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    pass
    #tools_IO.remove_files(folder_out,'*.png,*.gif')
    #data = G.get_data_IMDB('./data/ex_datasets/dataset_IMDB.csv')
    #data = G.get_data_facebook('./data/ex_GAD/facebook_combined.txt')

    # G.draw_graph(G.get_data_facebook('./data/ex_GAD/facebook_combined.txt'), filename_out='FB_shell.png',layout='shell', node_size=500,alpha=0.01)
    # G.draw_graph(G.get_data_facebook('./data/ex_GAD/facebook_combined.txt'), filename_out='FB_shell_sorted.png',layout='shell_sorted', node_size=500, alpha=0.01)
    # G.draw_graph(G.get_data_facebook('./data/ex_GAD/facebook_combined.txt'), filename_out='FB_sprng.png',layout='sprng', node_size=100,alpha=0.3)
    #G.draw_mat(G.get_data_facebook('./data/ex_GAD/facebook_combined.txt'), 'FB_mat_sorted.png', layout='shell_sorted')
    # G.draw_mat(G.get_data_facebook('./data/ex_GAD/facebook_combined.txt'), 'FB_mat.png', layout='shell')

    # G.draw_graph(G.get_data_karate(),filename_out='karate_shell.png',layout='shell',node_size=800)
    # G.draw_graph(G.get_data_karate(),filename_out='karate_shell_sorted.png',layout='shell_sorted',node_size=800)
    # G.draw_graph(G.get_data_karate(),filename_out='karate_sprng.png',layout='sprng',node_size=800)
    # G.draw_mat(G.get_data_karate(), 'karate_mat_sorted.png',layout='shell_sorted')
    # G.draw_mat(G.get_data_karate(), 'karate_mat.png', layout='shell')
    #tools_animation.folder_to_animated_gif_imageio(folder_out, folder_out+'FB2.gif', mask='*.png,*.jpg', framerate=10,stop_ms=4000,do_reverce=True)