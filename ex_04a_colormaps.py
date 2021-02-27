import cv2
import numpy
# ---------------------------------------------------------------------------------------------------------------------
import tools_image
import matplotlib.pyplot as plt
from matplotlib import cm
# ---------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_draw_numpy
folder_out = './data/output/'
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

    W = 640
    N = 32
    image = numpy.full((N,W,3), 0, dtype=numpy.uint8)
    colors = tools_draw_numpy.get_colors(N, colormap='gray')

    for n in range(N - 1):
        image[-50:,int(n * W / (N - 1)):int((n + 1) * W / (N - 1))] = colors[n]

    tools_IO.remove_files(folder_out)
    image = tools_image.desaturate_2d(image)
    for cm_name in cmap_names:
        res = tools_image.hitmap2d_to_colormap(image, plt.get_cmap(cm_name))
        cv2.imwrite(folder_out + '%s.png' % cm_name, res)
    return
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    ex_bars()
