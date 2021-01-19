import numpy
import seaborn
import matplotlib.pyplot as plt
import squarify
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
def ex_squarify_plot(df,idx_weights,idx_labels):

    col_label = df.columns.to_numpy()[idx_labels]
    col_weights = df.columns.to_numpy()[idx_weights]
    df_agg = df.groupby(col_label).sum()

    weights = df_agg.loc[:, col_weights].to_numpy()
    labels = df_agg.index.to_numpy()

    cmap = plt.get_cmap('Set2')
    colors = numpy.array([cmap(i / float(len(labels))) for i in range(len(labels))])

    plt.figure(figsize=(12, 4))
    squarify.plot(sizes=weights, label=labels, color=colors)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(folder_out + 'squarify.png')

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    df = seaborn.load_dataset('flights')
    ex_squarify_plot(df, idx_weights=2, idx_labels=0)