import numpy
import tools_IO
import tools_ML
import matplotlib.pyplot as plt
from scipy import stats
# ----------------------------------------------------------------------------------------------------------------------
ML = tools_ML.tools_ML(None)
# ----------------------------------------------------------------------------------------------------------------------
has_header,has_labels_first_col = True, True
folder_in = 'data/ex_pos_neg_football6/'
# ----------------------------------------------------------------------------------------------------------------------
def preprocess_array(X):
    #Y = numpy.array([x for x in X if x>0])
    Y = X
    return Y
# ----------------------------------------------------------------------------------------------------------------------
def analyze_pos_neg(filename_data_pos,filename_data_neg,folder_out):

    tools_IO.remove_files(folder_out,create=True)

    f_handle = open(folder_out + "descript.ion", "w+")
    f_handle.close()

    data_pos = tools_IO.load_mat(filename_data_pos, numpy.chararray, '\t')
    data_neg = tools_IO.load_mat(filename_data_neg, numpy.chararray, '\t')
    header, first_col, x_pos = ML.preprocess_header(data_pos, has_header, has_labels_first_col)
    header, first_col, x_neg = ML.preprocess_header(data_neg, has_header, has_labels_first_col)

    for i, feature in enumerate(header):
        x_p = x_pos[:, i]
        x_n = x_neg[:, i]

        A = preprocess_array(x_p)
        B = preprocess_array(x_n)

        plt.hist(A, bins='auto', alpha=0.5, label='pos', color='darkred')
        plt.hist(B, bins='auto', alpha=0.5, label='neg', color='darkgray')

        if len(A)>0 and len(B)>0:
            st = stats.ks_2samp(A, B)[0]
        else:
            st =0

        plt.legend(loc='upper right')
        name = feature.split()[0]
        name = name.split(':')[0]
        filename_out = '%03d_%s.png' % (i, name)
        plt.savefig(folder_out + filename_out)
        f_handle = open(folder_out + "descript.ion", "a+")
        f_handle.write("%s %f\n" % (filename_out, st))
        f_handle.close()
        plt.clf()

    return
# ----------------------------------------------------------------------------------------------------------------------
def analyze_train_test(filename_train,filename_test,folder_out):

    f_handle = open(folder_out + "descript.ion", "w+")
    f_handle.close()

    data_train = tools_IO.load_mat(filename_train, numpy.chararray, '\t')
    data_test  = tools_IO.load_mat(filename_test , numpy.chararray, '\t')
    header, y_train, x_train = ML.preprocess_header(data_train, has_header, has_labels_first_col)
    header, y_test , x_test  = ML.preprocess_header(data_test , has_header, has_labels_first_col)
    y_train = numpy.array(y_train,dtype=numpy.float)
    y_test  = numpy.array(y_test ,dtype=numpy.float)

    for i, feature in enumerate(header):
        x_t_p = x_train[y_train >0]
        x_t_n = x_train[y_train<=0]
        x_v_p = x_test [y_test  > 0]
        x_v_n = x_test [y_test <= 0]



        if len(x_t_p[:, i])>0 and len(x_v_p[:, i])>0:
            st = stats.ks_2samp(preprocess_array(x_t_p[:, i]), preprocess_array(x_v_p[:, i]))[0]
        else:
            st =0
        plt.hist(preprocess_array(x_t_p[:, i]), bins='auto', alpha=0.5, label='train pos', color='darkred')
        plt.hist(preprocess_array(x_v_p[:, i]), bins='auto', alpha=0.5, label='test pos', color='red')
        plt.legend(loc='upper right')
        name = feature.split()[0]
        name = name.split(':')[0]
        filename_out = 'pos_%03d_%s.png' % (i, name)
        plt.savefig(folder_out + filename_out)
        f_handle = open(folder_out + "descript.ion", "a+")
        f_handle.write("%s %f\n" % (filename_out, st))
        f_handle.close()
        plt.clf()


        if len(x_t_n[:, i])>0 and len(x_v_n[:, i])>0:
            st = stats.ks_2samp(preprocess_array(x_t_n[:, i]), preprocess_array(x_v_n[:, i]))[0]
        else:
            st =0
        plt.hist(preprocess_array(x_t_n[:, i]), bins='auto', alpha=0.5, label='train neg', color='darkgray')
        plt.hist(preprocess_array(x_v_n[:, i]), bins='auto', alpha=0.5, label='test neg', color='gray')
        plt.legend(loc='upper right')
        filename_out = 'neg_%03d_%s.png' % (i, name)
        plt.savefig(folder_out + filename_out)
        f_handle = open(folder_out + "descript.ion", "a+")
        f_handle.write("%s %f\n" % (filename_out, st))
        f_handle.close()
        plt.clf()

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    folder_out = 'data/output/'
    filename_data_pos,filename_data_neg = folder_in + 'pos.txt', folder_in + 'neg.txt'
    filename_train, filename_test = folder_in + 'train.txt', folder_in + 'test.txt'

    analyze_pos_neg(filename_data_pos,filename_data_neg,folder_out)
    #analyze_train_test(filename_train,filename_test,folder_out)

