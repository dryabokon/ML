import numpy
import seaborn
import matplotlib.pyplot as plt
import squarify
import pandas as pd
# ----------------------------------------------------------------------------------------------------------------------
import tools_draw_numpy
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
def ex_squarify_plot(df,idx_weights,idx_labels):

    col_label = df.columns.to_numpy()[idx_labels]
    col_weights = df.columns.to_numpy()[idx_weights]
    df_agg = df.groupby(col_label).sum()

    weights = df_agg.loc[:, col_weights].to_numpy()
    labels = df_agg.index.to_numpy()

    colors2 = (tools_draw_numpy.get_colors(1+len(labels),colormap = 'nipy_spectral',alpha_blend=0.25,shuffle=False)/255)[1:]
    colors2 = numpy.hstack((colors2,numpy.full((len(labels),1),1)))


    figure_squarify = plt.figure(figsize=(12, 4))
    squarify.plot(sizes=weights, label=labels, color=colors2)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(folder_out + 'squarify.png')

    return
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
# ----------------------------------------------------------------------------------------------------------------------
def ex_users():
    df = pd.read_csv(folder_in + 'users.csv', delimiter=',')
    df['user_id'] = 1
    ex_squarify_plot(df, idx_weights=0, idx_labels=3)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_flights():
    df = seaborn.load_dataset('flights')
    ex_squarify_plot(df, idx_weights=2, idx_labels=0)
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    #ex_flights()
    ex_users()