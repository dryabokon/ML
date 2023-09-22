import json
import pandas as pd
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output,dash_table
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
    #html.Div(id='user-table-container'),
    html.Div(id='selected-cell-attribute'),
    html.Div(id='selected-cell-data', style={'display': 'none'}),  # Hidden div to store selected cell's data
    dash_table.DataTable(
        id='user-table',
        columns=[
            {'name': 'Name', 'id': 'Name'},
            {'name': 'Email', 'id': 'Email'},
        ],
        # style_table={'overflowX': 'scroll'},
        # row_selectable='single',  # Allow selecting one row at a time
    )
])
# ----------------------------------------------------------------------------------------------------------------------
@app.callback(
    [Output('user-table', 'data'), Output('selected-cell-data', 'children')],
    [Input('user-search', 'value'), Input('user-table', 'selected_cells')]
)
def update_user_table(search_term, selected_cells):
    if search_term is None:
        return [], None
    else:
        filtered_data = filter_users(search_term)
        table_data = filtered_data.to_dict('records')

        selected_data = None
        if selected_cells:
            selected_row = selected_cells[0]['row']
            if 0 <= selected_row < len(table_data):
                selected_data = table_data[selected_row]

        return table_data, json.dumps(selected_data)
# ----------------------------------------------------------------------------------------------------------------------
@app.callback(
    Output('selected-cell-attribute', 'children'),
    Input('selected-cell-data', 'children')
)
def display_selected_cell_attribute(selected_data_json):
    if selected_data_json is None:
        return ''
    selected_data = json.loads(selected_data_json)
    if selected_data:
        return f'Selected User: {selected_data["Name"]}, Email: {selected_data["Email"]}'
    return ''
# ----------------------------------------------------------------------------------------------------------------------
def filter_users(search_term):
    df = pd.DataFrame(users_data)
    if len(search_term)==0:
        return df
    else:
        return df[df['Name'].str.contains(search_term, case=False)]
# ----------------------------------------------------------------------------------------------------------------------
def create_user_table(data):
    table = dash_table.DataTable(
        id='user-table',
        columns=[
            {'name': 'Name', 'id': 'Name'},
            {'name': 'Email', 'id': 'Email'},
        ],
        data=data.to_dict('records'),
        style_table={'overflowX': 'scroll'},
        row_selectable='single',  # Allow selecting one row at a time
    )
    return table
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
