import pandas as pd
# ---------------------------------------------------------------------------------------------------------------------
import tools_DF
import tools_plot_dancing
# ---------------------------------------------------------------------------------------------------------------------
folder_out = './data/output/'
# ---------------------------------------------------------------------------------------------------------------------
PD = tools_plot_dancing.Plotter_dancing(folder_out,dark_mode=False)
# ---------------------------------------------------------------------------------------------------------------------
#df = pd.read_csv('./data/ex_vacancies/bak/dou.csv',sep='\t')
# ---------------------------------------------------------------------------------------------------------------------
df = pd.read_csv('./data/ex_vacancies/input_02_yearly_hack.csv',sep='\t')
df = tools_DF.from_multi_column(df,idx_time=0)
# ---------------------------------------------------------------------------------------------------------------------
# df = pd.read_csv('./data/ex_population/population2.csv')
# df = df[df['time']>=1800]
# ---------------------------------------------------------------------------------------------------------------------
def ex01_static_timeline():
    PD.plot_static_timeline_chart(df, filename_out='EDA_plot_static_timeline_chart.png',to_ratios=False)
    return
# ---------------------------------------------------------------------------------------------------------------------
def ex02_plot_stacked_data():
    df2 = tools_DF.impute_na(df)

    PD.plot_stacked_data(df2,top_objects=10,filename_out='stacked.png',in_format_x='%Y',out_format_x='%Y',legend='DOU Top')
    return
# ---------------------------------------------------------------------------------------------------------------------
def ex03_dynamic_timeline():
    df_dynamic = PD.get_dynamics_data(df, col_time='time', col_label='label', col_value='value')
    PD.plot_dynamics_histo(df_dynamic, col_time='time', col_label='label', col_value='value',in_format_x='%Y',out_format_x='%Y',n_tops=12, n_extra=6)
    return
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    #ex01_static_timeline()
    #ex02_plot_stacked_data()
    ex03_dynamic_timeline()
