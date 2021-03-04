import numpy
import seaborn
import matplotlib.pyplot as plt
import pandas as pd
# ----------------------------------------------------------------------------------------------------------------------
import tools_draw_numpy
import tools_plot_v2
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
P = tools_plot_v2.Plotter(folder_out,dark_mode=True)
# ----------------------------------------------------------------------------------------------------------------------
def ex_squarify_plot(df,idx_weights,idx_labels):

    col_label = df.columns.to_numpy()[idx_labels]
    col_weights = df.columns.to_numpy()[idx_weights]
    df_agg = df.groupby(col_label).sum()

    weights = df_agg.loc[:, col_weights].to_numpy()
    labels = df_agg.index.to_numpy()

    P.plot_squarify(weights,labels,'squarify.png')
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_pie_plot(df,idx_weights,idx_labels):
    col_label = df.columns.to_numpy()[idx_labels]
    col_weights = df.columns.to_numpy()[idx_weights]
    df_agg = df.groupby(col_label).sum()

    weights = df_agg.loc[:, col_weights].to_numpy()
    labels = df_agg.index.to_numpy()

    colors = (tools_draw_numpy.get_colors(1 + len(labels), colormap='tab20', alpha_blend=0.25, shuffle=True) / 255)[1:]
    colors = numpy.hstack((colors, numpy.full((len(labels), 1), 1)))

    idx = numpy.argsort(-weights)
    fig= plt.figure()
    ax = fig.gca()
    ax.pie(weights[idx],  labels=labels[idx], autopct='%1.1f%%',shadow=False, startangle=90,counterclock=False,colors=colors, pctdistance=0.5)
    plt.savefig(folder_out + 'pie.png')

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_users():
    df = pd.read_csv(folder_in + 'users.csv', delimiter=',')
    df['user_id'] = 1
    ex_squarify_plot(df, idx_weights=0, idx_labels=3)
    ex_pie_plot(df, idx_weights=0, idx_labels=3)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_flights():
    df = seaborn.load_dataset('flights')
    ex_squarify_plot(df, idx_weights=2, idx_labels=0)
    ex_pie_plot(df, idx_weights=2, idx_labels=0)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_ML_areas():
    df = pd.read_csv(folder_in + 'ML_areas.csv', delimiter=',')
    ex_squarify_plot(df, idx_weights=0, idx_labels=1)
    ex_pie_plot(df, idx_weights=0, idx_labels=1)
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    #ex_flights()
    #ex_users()
    ex_ML_areas()