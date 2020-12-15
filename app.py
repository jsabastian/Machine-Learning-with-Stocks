
import plotly.express as px

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

test = pd.read_csv('test_data/AAPL_test_data.csv')
test2= pd.DataFrame(test)
test3=test2["close"]

fig = px.scatter(test3)


app.layout = html.Div(children=[
  html.H1(children = ('Stock Prediction')),

  html.Div(children = '''
    Cyber-Booleans: For all mankind
    '''),
    
    dcc.Graph(
      id='test',
      figure = fig
    
    )


])

if __name__ == '__main__':
    app.run_server(debug=False)