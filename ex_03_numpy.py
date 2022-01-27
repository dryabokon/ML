import tools_IO

import numpy
import pickle


# ----------------------------------------------------------------------------------------------------------------------
def ex_01_create():
    A = numpy.zeros(4)

    B = numpy.empty((3, 2))

    C = numpy.ones((2, 2))

    D = numpy.zeros((2, 3))

    E = numpy.full((2, 3), 1)

    F = numpy.eye(4)

    G = numpy.full((320, 240), 32, dtype=numpy.uint8)

    H = numpy.full((320, 240, 3), 32, dtype=numpy.uint8)

    I = numpy.full((320, 240, 3), (0, 0, 200), dtype=numpy.uint8)

    J = numpy.array((1, 2, 3))

    K = numpy.array((('00', '01', '02'), ('10', '11', '12')))

    L = numpy.linspace(10, 25, 9)

    M = numpy.arange(10, 25, 5)

    return


# ----------------------------------------------------------------------------------------------------------------------

def ex_02_inspect():
    A = numpy.full((2, 3), 1)

    sh = A.shape
    ndims = A.ndim
    size = A.size
    the_type = A.dtype
    type_name = A.dtype.name
    L = len(A)

    return


# ----------------------------------------------------------------------------------------------------------------------
def ex_03_combine():
    A = numpy.full((2, 3), 1)
    B = numpy.full((5, 3), 3)
    C = numpy.full((2, 4), 2)
    D = numpy.full((2, 3), 4)

    K1 = numpy.vstack((A, B))
    L1 = numpy.hstack((A, C))

    K2 = numpy.append(A, B, axis=0)
    L2 = numpy.append(A, C, axis=1)

    K3 = numpy.concatenate((A, B), axis=0)
    L3 = numpy.concatenate((A, C), axis=1)

    P = numpy.array((A, D))

    return


# ----------------------------------------------------------------------------------------------------------------------
def ex_04_insert():
    A = numpy.full((2, 3), 'a')
    B = numpy.full((5, 3), 'b')
    C = numpy.full((2, 4), 'c')

    K = numpy.insert(A, [2], B, axis=0)
    L = numpy.insert(A, [1], C, axis=1)

    return


# ----------------------------------------------------------------------------------------------------------------------
def ex_05_delete():
    A = numpy.full((2, 3), 1)

    K = numpy.delete(A, [1], axis=0)
    L = numpy.delete(A, [1], axis=1)

    return


# ----------------------------------------------------------------------------------------------------------------------
def ex_06_reshape():
    A = numpy.full((10, 16, 3), 1)  # (10 x 16 x 3)

    B = numpy.swapaxes(A, 0, 1)  # (16 x 10 x 3)
    C = numpy.swapaxes(A, 0, 2)  # ( 3 x 16 x 10)
    D = numpy.swapaxes(A, 1, 2)  # (10 x  3 x 16)

    B2 = numpy.transpose(A, (1, 0, 2))  # (16 x 10 x 3)
    E = numpy.transpose(A, (2, 0, 1))  # (3  x 10 x 16)

    return


# ----------------------------------------------------------------------------------------------------------------------
def ex_07_slicing():
    A = numpy.full((10, 16, 3), 1)

    B1 = A[0:4]  # (4 x 16 x 3)
    B2 = A[:4]

    C1 = A[8:]  # (2 x 16 x 3)
    C2 = A[10 - 2:]
    C3 = A[-2:]

    D1 = A[:, :13]  # (10 x 13 x 3)
    D2 = A[:, -3]

    E = A[0, :, :]  # (16,3)
    F = A[:, 0, :]  # (10,3)
    G = A[:, :, 0]  # (10,16)

    return


# ----------------------------------------------------------------------------------------------------------------------

def ex_08_order():
    A = numpy.array(
        (('Apple ', 2, 4000),
         ('Lemon ', 3, 1000),
         ('Milk  ', 7, 2000),
         ('Banana', 9, 3000),
         ('Coffee', 7, 6000)))

    B_fail = numpy.sort(A, axis=0)
    C_fail = numpy.sort(A, axis=1)

    idx0 = numpy.argsort(A[:, 0])
    idx1 = numpy.argsort(A[:, 1])
    idx2 = numpy.argsort(A[:, 2])

    B = A[idx0]
    C = A[idx1]
    D = A[idx2]

    B2 = numpy.array(sorted(A, key=lambda A: A[0]))
    C2 = numpy.array(sorted(A, key=lambda A: A[1]))
    D2 = numpy.array(sorted(A, key=lambda A: A[2]))

    # print(B)
    # print()
    # print(C)
    # print()
    # print(D)

    return


# ----------------------------------------------------------------------------------------------------------------------

def ex_09_aggregates():
    A = numpy.array(
        (('Apple ', 2, 4000),
         ('Lemon ', 0, 0000),
         ('Milk  ', 0, 2000),
         ('Banana', 9, 3000),
         ('Coffee', 7, 6000))).astype(numpy.chararray)

    #AA = A.copy()
    #AA = A[:, [1, 2]].astype(numpy.chararray)
    AA = A[:, [1, 2]].astype(numpy.int)

    A_sum = numpy.sum(AA, axis=0)
    A_sum = numpy.sum(AA, axis=0)
    A_avg = numpy.mean(AA, axis=0)
    A_min = numpy.min(AA, axis=0)
    A_max = numpy.max(AA, axis=0)
    A_nzr = numpy.count_nonzero(AA, axis=0)

    return


# ----------------------------------------------------------------------------------------------------------------------
def ex_10_IO_bin_npy():
    A = numpy.array(
        (('Apple ', 2, 4000),
         ('Lemon ', 0, 0000),
         ('Milk  ', 0, 2000),
         ('Banana', 9, 3000),
         ('Coffee', 7, 6000)))

    AA = A[:, [1, 2]].astype(numpy.int)

    numpy.save('./A', A)
    B = numpy.load('./A.npy')


    return


# ----------------------------------------------------------------------------------------------------------------------
def ex_10_IO_bin_pickle():
    A = numpy.array(
        (('Apple ', 2, 4000),
         ('Lemon ', 0, 0000),
         ('Milk  ', 0, 2000),
         ('Banana', 9, 3000),
         ('Coffee', 7, 6000)))

    AA = A[:, [1, 2]].astype(numpy.int)

    with open('./A.dat', "wb") as f:
        pickle.dump(A, f)

    with open('./A.dat', "rb") as f:
        B = pickle.load(f)

    return


# ----------------------------------------------------------------------------------------------------------------------
def ex_10_IO_text():
    A = numpy.array(
        (('Apple ', 2, 4000),
         ('Lemon ', 0, numpy.nan),
         ('Milk  ', 0, 2000),
         ('Banana', 9, 3000),
         ('Coffee', 7, 6000)))

    data_type = A.dtype

    numpy.savetxt('A.txt', A, fmt='%s', delimiter='\t')
    B = numpy.loadtxt('A.txt', dtype=data_type, delimiter='\t')
    return


# ----------------------------------------------------------------------------------------------------------------------
def ex_11_copies():
    A = numpy.array(
        (('Apple ', 2, 4000),
         ('Lemon ', 0, numpy.nan),
         ('Milk  ', 0, 2000),
         ('Banana', 9, 3000),
         ('Coffee', 7, 6000)))


    B = A.copy()
    C = A
    #

    B[0, 0] = 'Orange'  # Updates B
    C[0, 0] = 'Peach'  # Updates both A and C (!!)

    return


# ----------------------------------------------------------------------------------------------------------------------
def ex_12_ravel():
    A = numpy.array(
        (('Apple ', 2, 4000),
         ('Lemon ', 3, numpy.nan),
         ('Milk  ', 7, 2000),
         ('Banana', 9, 3000),
         ('Coffee', 7, 6000)))

    F1 = numpy.ravel(A.copy())
    F2 = A.flatten()

    idx_cr = numpy.unravel_index([2, 4, 5], A.shape)

    A[idx_cr] = numpy.nan
    #idx_cr_custom  = numpy.array([[0, 2], [1, 1], [1, 2]])
    A[idx_cr] = numpy.nan

    # print(A)
    return


# ----------------------------------------------------------------------------------------------------------------------
def ex_13_printoptions():
    A = numpy.array([[1.00002]])
    print(A)

    numpy.set_printoptions(precision=3)
    print(A)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_14_nan():
    A = numpy.zeros((2, 2))
    A[0, 0] = numpy.nan

    mask_is_nan = numpy.isnan(A)
    A_has_any_nan = numpy.any(numpy.isnan(A))

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    ex_01_create()
    # ex_02_inspect()
    # ex_03_combine()
    # ex_04_insert()
    # ex_05_delete()
    # ex_06_reshape()
    # ex_07_slicing()
    # ex_08_order()
    # ex_09_aggregates()
    # ex_10_IO_bin_npy()
    # ex_10_IO_bin_pickle()
    # ex_10_IO_text()
    # ex_11_copies()
    # ex_12_ravel()
    # ex_13_printoptions()
    #

    #ex_10_IO_bin_npy()
