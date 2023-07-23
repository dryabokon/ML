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
def plot_bars_CV(image,values,vmin=None,vmax=None,width=0.7,pad=0.1):
    if vmin is None:
        vmin=numpy.min(values)
    if vmax is None:
        vmax = numpy.max(values)

    H,W = image.shape[:2]
    x_pos = numpy.linspace(W*pad,W*(1-pad),values.shape[0]+1)

    d = 0.5*W*(1-2*pad)*width/(values.shape[0])
    y_pos = [H*pad+(H*(1-pad)-H*pad)*(v-vmin)/(vmax-vmin) for v in values]
    rects = numpy.array([[[(x1+x2)/2-d,H*(1-pad)],[(x1+x2)/2+d,H-y]] for x1,x2,y in zip(x_pos[:-1],x_pos[1:],y_pos)])

    image = tools_draw_numpy.draw_rects(image, rects,colors=(100,0,0))

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

    tools_IO.remove_files(folder_out,create=True)

    P.empty(filename_out='xxx.png')
    #ex_titatic_pairplot()
    #ex_iris_pairplot()

    # image = numpy.full((240,320,3),128,dtype=numpy.uint8)
    # values = numpy.array([128,50,20,30])
    # image = plot_bars_CV(image,values,vmin=0,vmax=255,width=0.7)
    # cv2.imwrite(folder_out+'histo.png',image)

