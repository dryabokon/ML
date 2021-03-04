import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from io import BytesIO
import base64
from sklearn.datasets import make_regression
from plotly.tools import mpl_to_plotly
# ----------------------------------------------------------------------------------------------------------------------
import tools_plot_v2
# ----------------------------------------------------------------------------------------------------------------------
P = tools_plot_v2.Plotter(dark_mode=True)
# ----------------------------------------------------------------------------------------------------------------------
def fig_to_uri(in_fig, close_all=True, **save_args):
    out_img = BytesIO()
    in_fig.savefig(out_img, format='png', **save_args)
    if close_all:
        in_fig.clf()
        plt.close('all')
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/png;base64,{}".format(encoded)
# ----------------------------------------------------------------------------------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
# ----------------------------------------------------------------------------------------------------------------------
def get_data():
    X, Y = make_regression(n_samples=100, n_features=2, noise=50.0)
    Y[Y <= 0] = 0
    Y[Y > 0] = 1
    return X,Y
# ----------------------------------------------------------------------------------------------------------------------
X, Y = get_data()
# ----------------------------------------------------------------------------------------------------------------------
fig = mpl_to_plotly(P.plot_2D_features_multi_Y(X, Y))
# ----------------------------------------------------------------------------------------------------------------------
app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),
    html.Div(children='''Dash: A web application framework for Python.'''),
    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=False)