import os
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
import tools_IO
# ---------------------------------------------------------------------------------------------------------------------
import generator_Bay
import generator_Gauss
import generator_Other
import generator_Manual
# ---------------------------------------------------------------------------------------------------------------------
def generate_data_syntetic(folder_output,dim = 2):
    if not os.path.exists(folder_output):
        os.makedirs(folder_output)
    else:
        tools_IO.remove_files(folder_output)

    #Generator = generator_Bay.generator_Bay(dim)
    #Generator = generator_Gauss.generator_Gauss(dim)
    Generator = generator_Other.generator_Other(dim)

    Generator.create_pos_neg_samples(folder_output + 'data_pos.txt', folder_output + 'data_neg.txt')
    tools_IO.plot_2D_samples_from_folder(folder_output,  add_noice=1)
    plt.show()
    return

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #folder_out = 'data/output/'
    folder_out = 'data/ex06/'
    generate_data_syntetic(folder_out)
