import numpy
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
# ----------------------------------------------------------------------------------------------------------------------
import tools_GIS
import tools_draw_numpy
import tools_plot_v2
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_GIS/'
folder_out = './data/output/'
filename_in = folder_in + 'solidario_filtered.csv'
# ----------------------------------------------------------------------------------------------------------------------
G = tools_GIS.tools_GIS(folder_out=folder_out)
P = tools_plot_v2.Plotter(folder_out=folder_out)
colors255 = tools_draw_numpy.get_colors(256,colormap = 'jet')
clr_background = numpy.array((32,32,32))
# ----------------------------------------------------------------------------------------------------------------------
def remove_garmage_timestamps(df):
    keyword = '(Coordinated Universal Time)'

    for r in range(df.shape[0]):
        for c in range(df.shape[1]):
            record = df.iloc[r,c]
            if isinstance(record,str):
                if record.find(keyword)!=-1:
                    df.iloc[r, c] = '-'

    return df
# ----------------------------------------------------------------------------------------------------------------------
def ex_airport():
    df = pd.read_csv(folder_in + 'airports.csv', sep='\t')
    idx_lat, idx_long = 2, 3
    image = G.draw_points(df, idx_lat=idx_lat, idx_long=idx_long)
    cv2.imwrite(folder_out + 'airports_points.png', image)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_solidario():
    df = pd.read_csv(folder_in + 'solidario.csv', sep='\t')
    idx_lat, idx_long = 3, 4
    idx_value = 25

    idx =numpy.where(df['Exposure Flood'].values ==0)[0]

    df.drop(idx, 0, inplace=True)

    image = G.draw_points(df, idx_lat=idx_lat, idx_long=idx_long,idx_value=idx_value,draw_terrain=False)
    cv2.imwrite(folder_out + 'solidario_points.png', image)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_interpolate():
    def func(x, y):
        return x * (1 - x) * numpy.cos(4 * numpy.pi * x) * numpy.sin(4 * numpy.pi * y ** 2) ** 2

    XY = numpy.random.random((18, 2))
    Z = func(XY[:, 0], XY[:, 1])
    Z -= Z.min()
    Z /= Z.max() / 255
    N = 100
    method = 'linear'
    grid_x, grid_y = numpy.meshgrid(numpy.linspace(0, 1, num=N), numpy.linspace(0, 1, num=N))
    grid_z = griddata(XY, Z, (grid_x, grid_y), method=method)
    plt.imshow(grid_z, extent=(0, 1, 0, 1),origin='lower')
    colors255 = (tools_draw_numpy.get_colors(256, colormap='viridis', shuffle=False))[:, [2, 1, 0]].astype(numpy.float32) / 255
    colors = [colors255[int(z)] for z in Z]
    plt.scatter(x=XY[:, 0], y=XY[:, 1], color=colors, s=100, alpha=1, edgecolor=(0.25, 0.25, 0.25, 1))
    plt.show()
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_interpolate2():
    df0 = pd.read_csv(folder_in + 'solidario.csv', sep='\t')
    idx_lat, idx_long = 3, 4
    df0 = df0.iloc[:, [idx_lat, idx_long]]
    df0 = df0.dropna()
    df = df0.drop_duplicates()


    numpy.random.seed(6)
    idx = numpy.random.choice(df.shape[0],13,replace=False)
    df = df.iloc[idx,:]
    df['value'] = numpy.random.random((df.shape[0], 1))

    image = G.draw_points(df, idx_lat=0, idx_long=1, idx_value=2, draw_terrain=False)
    cv2.imwrite(folder_out + 'solidario_points.png', image)

    N = 100
    XY = df.iloc[:, :2].values
    Z = df.iloc[:, 2].values

    #method = 'linear'
    #method = 'nearest'
    method = 'cubic'
    grid_x, grid_y = numpy.meshgrid(numpy.linspace(XY[:,0].min(), XY[:,0].max(), num=N), numpy.linspace(XY[:,1].min(), XY[:,1].max(), num=N))
    grid_z = griddata(XY, Z, (grid_x, grid_y), method=method)
    idx = numpy.argwhere(~numpy.isnan(grid_z.flatten())).flatten()
    df2=pd.DataFrame({'X':grid_x.flatten()[idx],'Y':grid_y.flatten()[idx],'Z':grid_z.flatten()[idx]})
    image = G.draw_points(df2, idx_lat=0, idx_long=1, idx_value=2, draw_terrain=False,edgecolor=None)
    cv2.imwrite(folder_out + 'solidario_points2.png', image)

    # df.to_csv(folder_out+'df.txt',index=False,sep='\t')
    # df2.to_csv(folder_out + 'df2.txt', index=False,sep='\t')

    return
# ----------------------------------------------------------------------------------------------------------------------
def generate_temparature_ts(N):

    t_min = 5
    t_max = 35

    t = numpy.arange(0, N, 1)

    Y = (t_min+t_max)/2 + (t_max-t_min)/2*numpy.sin(2*numpy.pi*(t-N/4)/N)
    res = numpy.concatenate(([t],[Y]),axis=0).T

    return res
# ----------------------------------------------------------------------------------------------------------------------
def get_temp_color(temperature,t_min=-20,t_max=40):
    if temperature<t_min:
        return colors255[0]
    if temperature>=t_max:
        return colors255[-1]
    color  = colors255[int(255*(temperature-t_min)/(t_max-t_min))]

    return color
# ----------------------------------------------------------------------------------------------------------------------
def map_to_chart_Y(temperature,image):
    value = int(image.shape[0] / 2 - temperature)
    if value<0:
        value =0
    if value>=image.shape[0]:
        value = image.shape[0]-1
    return value
# ----------------------------------------------------------------------------------------------------------------------
def draw_TS(image,pointsXY,probability,clr_background):
    for p in pointsXY:
        time = int(p[0])
        temperature = p[1]
        color = get_temp_color(temperature)
        alpha_blend = 1-probability

        color = ((alpha_blend) * numpy.array(clr_background) + (1 - alpha_blend) * numpy.array(color))

        valueY = map_to_chart_Y(temperature,image)
        image[valueY, time, :]=color
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    # pointsXY = generate_temparature_ts(365)
    # df = pd.DataFrame({'time':pointsXY[:,0],'Nature_temp':pointsXY[:,1]})
    #P.TS_matplotlib(df,idxs_target=1,idx_feature=None,idxs_fill=None,remove_legend=False,x_range=None,y_range=[-20,50],palette='tab10',figsize=(19, 3),filename_out='TS.png')
    #P.TS_seaborn(df, idxs_target=1, idx_feature=None, mode='scatterplot', remove_legend=False,remove_xticks=True,x_range=[0,df.shape[0]], major_step=30, palette='tab10', transparency=0.8, figsize=(18, 3), filename_out='TS.png')


    image = numpy.full((180,2*365,3),clr_background,dtype=numpy.uint8)
    temparature_natural  = generate_temparature_series(image.shape[1])
    probability = 0.7
    draw_TS(image, temparature_natural, probability,clr_background)

    cv2.imwrite(folder_out + 'nature.png', image)