import tools_plotly
import tools_plotly_draw
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
Builder = tools_plotly.Plotly_builder(folder_out)
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    Builder.run_server()