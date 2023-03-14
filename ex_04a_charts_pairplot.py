import cv2
import numpy
import seaborn
import pandas as pd
# ----------------------------------------------------------------------------------------------------------------------
import tools_DF
import tools_plot_v2
import tools_IO
import tools_Hyptest
import tools_draw_numpy
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
P = tools_plot_v2.Plotter(folder_out,dark_mode=False)
HT = tools_Hyptest.HypTest()
# ----------------------------------------------------------------------------------------------------------------------
def ex_heart():
    df = pd.read_csv(folder_in + 'dataset_heart.csv', delimiter=',')
    P.pairplots_df(df, add_noise=True,idx_target=-1)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_houses():
    df = pd.read_csv(folder_in + 'dataset_kc_house_data.csv', delimiter=',')
    P.pairplots_df(df, idx_target=-1)
    return
# ----------------------------------------------------------------------------------------------------------------------
def plot_bars_CV(image,values,vmin=None,vmax=None):
    if vmin is None:
        vmin=numpy.min(values)
    if vmax is None:
        vmax = numpy.max(values)

    H,W = image.shape[:2]
    x_pos = numpy.linspace(0,W,values.shape[0]+1)

    y_pos = [0+H*(v-vmin)/(vmax-vmin) for v in values]
    rects = numpy.array([[[x1,H-1],[x2,H-y]] for x1,x2,y in zip(x_pos[:-1],x_pos[1:],y_pos)])

    image = tools_draw_numpy.draw_rects(image, rects, color=(0,0,200), w=2, alpha_transp=0.8)

    return image
# ----------------------------------------------------------------------------------------------------------------------
def ex_titatic_histo():
    df = seaborn.load_dataset('titanic')
    #df.to_csv(folder_out + 'titanic.csv', index=False)
    df = tools_DF.impute_na(df,strategy='mean')

    df = df.drop(columns=['alive'])
    df['survived'] = df['survived'].map({0: '0-not survived', 1: '1-survived'})
    P.set_color('0-not survived',P.color_red)
    P.set_color('1-survived',P.color_blue)

    idx_target = df.columns.get_loc('survived')
    #P.histoplots_df(df, idx_target,transparency=0.5)

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_titatic_pairplot():
    df = seaborn.load_dataset('titanic')
    df = tools_DF.impute_na(df,strategy='mean')

    df = df.drop(columns=['alive'])

    df['survived'] = df['survived'].map({0: '0-not survived', 1: '1-survived'})
    P.set_color('0-not survived', P.color_blue)
    P.set_color('1-survived', P.color_amber)
    idx_target = df.columns.get_loc('survived')



    P.pairplots_df(df, idx_target, None, add_noise=False,remove_legend=True)

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_iris_pairplot():
    df,idx_target = pd.read_csv(folder_in + 'dataset_iris.csv'),0
    df.iloc[:,0] = df.iloc[:,0].map({0:'type A',1:'type B',2:'type C'})

    P.set_color('type A', P.color_aqua)
    P.set_color('type B', P.color_marsala)
    P.set_color('type C', P.color_sky)

    P.pairplots_df(df, idx_target)

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #tools_IO.remove_files(folder_out,create=True)

    #ex_titatic_pairplot()
    #ex_iris_pairplot()

    image = numpy.full((240,320,3),255,dtype=numpy.uint8)
    values = numpy.array([128,50,20,30])
    image = plot_bars_CV(image,values,vmin=0,vmax=255)
    cv2.imwrite(folder_out+'histo.png',image)

