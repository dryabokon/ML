import cv2
import numpy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
# ---------------------------------------------------------------------------------------------------------------------
import tools_image
import tools_IO
import tools_draw_numpy
import tools_plot_v2
# ---------------------------------------------------------------------------------------------------------------------
folder_out = './data/output/'
# ---------------------------------------------------------------------------------------------------------------------
P = tools_plot_v2.Plotter(folder_out)
# ---------------------------------------------------------------------------------------------------------------------
#cmap_names =['viridis', 'plasma', 'inferno', 'magma', 'cividis']
#cmap_names =['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
#cmap_names =['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink','spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia','hot', 'afmhot', 'gist_heat', 'copper']
#cmap_names =['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu','RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
#cmap_names = ['twilight', 'twilight_shifted', 'hsv']
#cmap_names = ['Pastel1', 'Pastel2', 'Paired', 'Accent','Dark2', 'Set1', 'Set2', 'Set3','tab10', 'tab20', 'tab20b', 'tab20c']
cmap_names = ['flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern','gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg','gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']
# ---------------------------------------------------------------------------------------------------------------------
def ex_circles():
    W,H = 800,600
    N =10
    image = numpy.full((H, W, 3), 0, dtype=numpy.uint8)
    colors = tools_draw_numpy.get_colors(N,colormap='gray')
    for n in range(N):
        r = int(numpy.random.rand()*W)
        c = int(numpy.random.rand()*H)
        rad = int(H/10 + numpy.random.rand()*H/3)
        image = tools_draw_numpy.draw_circle(image, r, c, rad, colors[n])

    for n in range(N-1):
        image[int(n*H/(N-1)):int((n+1)*H/(N-1)),-50:]=colors[n]


    tools_IO.remove_files(folder_out)
    image = tools_image.desaturate_2d(image)
    for cm_name in cmap_names:
        res = tools_image.hitmap2d_to_colormap(image,plt.get_cmap(cm_name))
        cv2.imwrite(folder_out + 'res_%s.png'%cm_name,res)
    return
# ---------------------------------------------------------------------------------------------------------------------
def ex_bars():

    W,H = 640,32
    tools_IO.remove_files(folder_out)
    image = numpy.full((H,W,3), 0, dtype=numpy.uint8)

    for cm_name in cmap_names:
        N = plt.get_cmap(cm_name).N
        colors = tools_draw_numpy.get_colors(N, colormap=cm_name)
        for n in range(N - 1):
            image[:,int(n * W / (H - 1)):int((n + 1) * W / (H - 1))] = colors[n]

        cv2.imwrite(folder_out + '%s.png' % cm_name, image)
    return
# ---------------------------------------------------------------------------------------------------------------------
def ex_dark_light_scatter():
    X, Y = make_regression(n_samples=100, n_features=2, noise=50.0)
    Y[Y <= 0] = 0
    Y[Y > 0] = 1

    P = tools_plot_v2.Plotter(folder_out, dark_mode=False)
    df = pd.DataFrame({'Y':Y,'x1':X[:,0],'x2':X[:,1]})
    P.plot_2D_features(df, filename_out='seaborn_scatter_light.png')

    P = tools_plot_v2.Plotter(folder_out,dark_mode=True)
    P.plot_2D_features(df, filename_out='seaborn_scatter_dark.png')
    return
# ---------------------------------------------------------------------------------------------------------------------
def ex_dark_light_line():

    tpr = numpy.linspace(0,1,20)
    fpr = numpy.linspace(0.3,0.9,20)

    P = tools_plot_v2.Plotter(folder_out, dark_mode=False)
    P.plot_tp_fp(tpr, fpr, 0.5, caption='', filename_out='matplotlib_line_light.png')

    P = tools_plot_v2.Plotter(folder_out, dark_mode=True)
    P.plot_tp_fp(tpr, fpr, 0.5, caption='', filename_out='matplotlib_line_dark.png')

    return
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #ex_dark_light_line()
    #ex_dark_light_scatter()

    ex_bars()
