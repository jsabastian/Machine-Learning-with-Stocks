import plotly.express as px

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

input_CSV = pd.read_csv("test_data/Stock_Close_Predictions.csv")
Dash_df= pd.DataFrame(input_CSV)
Dash_df.set_index("Date", inplace=True, drop=True)

# fig = px.scatter(Dash_df["Actual Stock Price ($ USD)"])
fig = go.Figure()
fig.add_trace(go.Scatter(x=test2.index, y=test2["Actual Close Price ($ USD)"],
                    mode='scatter',
                    name='Actual Stock Price'))
fig.add_trace(go.Scatter(x=test2.index, y=test2["Predicted Close Price ($ USD)"],
                    mode='scatter',
                    name='Predicted Stock Close Price'))  
fig.show()

app.layout = html.Div(children=[
  html.H1(children = ('Stock Prediction')),

  html.Div(children = '''
    Cyber-Booleans: For All Mankind
    '''),
    
    dcc.Graph(
      id='test',
      figure = fig
    
    )
])

# app.layout = html.Div([
   
#     html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
#     dcc.Tabs(children=[
       
#         dcc.Tab(label='Stock Data',children=[
#             html.Div([
#                 html.H2("Actual Closing Price",style={"textAlign": "center"}),
#                 dcc.Graph(
#                     id="Actual Data",
#                     figure={
#                         "data":[
#                             go.Scatter(
#                                 x=train.index,
#                                 y=valid["Close"],
#                                 mode='markers'
#                             )
#                         ],
#                         "layout":go.Layout(
#                             title='scatter plot',
#                             xaxis={'title':'Date'},
#                             yaxis={'title':'Closing Rate'}
#                         )
#                     }
#                 ),
#                 html.H2("LSTM Predicted Closing Price",style={"textAlign": "center"}),
#                 dcc.Graph(
#                     id="Predicted Data",
#                     figure={
#                         "data":[
#                             go.Scatter(
#                                 x=valid.index,
#                                 y=valid["Predictions"],
#                                 mode='markers'
#                             )
#                         ],
#                         "layout":go.Layout(
#                             title='scatter plot',
#                             xaxis={'title':'Date'},
#                             yaxis={'title':'Closing Rate'}
#                         )
#                     }
#                 )                
#             ])                
#         ])

if __name__ == '__main__':
    app.run_server(debug=False)