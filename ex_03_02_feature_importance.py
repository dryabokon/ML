import numpy
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes,load_wine
from xgboost import XGBClassifier
from xgboost import plot_importance
# ----------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_plot
# ----------------------------------------------------------------------------------------------------------------------
def example_01():
    dataset = load_wine()
    X,Y = dataset.data, dataset.target
    model = XGBClassifier()
    model.fit(X, Y)

    feature_importances = model.get_booster().get_score()

    print(feature_importances)
    plot_importance(model)
    plt.show()

    return
# ----------------------------------------------------------------------------------------------------------------------
def example_02():
    has_header, has_labels_first_col = True, True
    folder_in = 'data/ex_pos_neg_apnea/'

    filename_data_pos = folder_in + 'pos.txt'
    filename_data_neg = folder_in + 'neg.txt'

    data_pos = tools_IO.load_mat(filename_data_pos, numpy.chararray, '\t')
    data_neg = tools_IO.load_mat(filename_data_neg, numpy.chararray, '\t')
    if has_header:
        header = numpy.array(data_pos[0, :], dtype=numpy.str)
        x_pos = data_pos[1:, :]
        x_neg = data_neg[1:, :]
    else:
        header = None
        x_pos = data_pos[:, :]
        x_neg = data_neg[:, :]

    if has_labels_first_col:
        x_pos = x_pos[:, 1:]
        x_neg = x_neg[:, 1:]
        if header is not None:
            header = header[1:]

    Pos = x_pos.shape[0]
    Neg = x_neg.shape[0]

    X = numpy.vstack((x_pos, x_neg)).astype(numpy.float)
    Y = numpy.hstack((numpy.full(Pos, 1), numpy.full(Neg, 0)))

    fig = plt.figure(figsize=(12, 6))
    tools_plot.plot_feature_importance(plt.subplot(1, 1, 1), fig, X, Y, header)
    plt.show()

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    example_02()
