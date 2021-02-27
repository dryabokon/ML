import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.express as px
# ----------------------------------------------------------------------------------------------------------------------
fig_iris = px.scatter(px.data.iris(), x="sepal_width", y="sepal_length", color="species")
# ----------------------------------------------------------------------------------------------------------------------
gapminder = px.data.gapminder()
fig_gapminder = px.scatter(gapminder.query("year==2007"), x="gdpPercap", y="lifeExp", size="pop", color="continent",hover_name="country", log_x=True, size_max=60)
# ----------------------------------------------------------------------------------------------------------------------
app = dash.Dash()
#app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})
# ----------------------------------------------------------------------------------------------------------------------
app.layout = html.Div([

    html.Label('Choose dataset for plotting!'),

    dcc.Dropdown(
        id='demo-dropdown',
        options=[
            {'label': 'gapminder', 'value': 'gm'},
            {'label': 'iris', 'value': 'ir'},
        ],
        value='gm',
        style={'width': '50%'
            }
    ),

    #html.Hr(),
    html.Div(id='res1'),

], className='container')
# ----------------------------------------------------------------------------------------------------------------------
@app.callback(dash.dependencies.Output('res1', 'children'),[dash.dependencies.Input('demo-dropdown', 'value')])
def update_output(value):
    if value=='gm':
        return dcc.Graph(id="graph", figure=fig_gapminder)
    else:
        return dcc.Graph(id="graph", figure=fig_iris)
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=False)