import pandas as pd
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output
# ----------------------------------------------------------------------------------------------------------------------
users_data = {
    'Name': ['John Smith', 'Jane Doe', 'Alice Johnson', 'Bob Brown', 'Charlie Davis'],
    'Email': ['john@example.com', 'jane@example.com', 'alice@example.com', 'bob@example.com', 'charlie@example.com'],
}
# ----------------------------------------------------------------------------------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
# ----------------------------------------------------------------------------------------------------------------------
app.layout = html.Div([
    html.H1("User Search App"),
    dbc.Input(type="text", id="user-search", placeholder="Search for a user..."),
    html.Div(id='user-table-container'),
    html.Table(id='user-table'),
])
# ----------------------------------------------------------------------------------------------------------------------
@app.callback(
    [Output('user-table', 'children')],
    [Input('user-search', 'value')]
)
def update_user_table(search_term):
    if search_term is None:
        return [html.Tr(html.Td("Enter a search term to find users."))]
    else:
        filtered_data = filter_users(search_term)
        table = create_user_table(filtered_data)
        return [table]
# ----------------------------------------------------------------------------------------------------------------------
def filter_users(search_term):
    df = pd.DataFrame(users_data)
    if len(search_term)==0:
        return df
    else:
        return df[df['Name'].str.contains(search_term, case=False)]
# ----------------------------------------------------------------------------------------------------------------------
def create_user_table(data):
    table_rows = []
    for index, row in data.iterrows():
        row_data = [html.Td(row['Name']), html.Td(row['Email'])]
        table_rows.append(html.Tr(row_data))

    table_header = [
        html.Th("Name"),
        html.Th("Email")
    ]

    table = dbc.Table([html.Thead(table_header), html.Tbody(table_rows)], bordered=True)
    return table
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
