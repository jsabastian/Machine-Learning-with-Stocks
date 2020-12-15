import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# Import CSV and re-build DataFrame using data
input_CSV = pd.read_csv("test_data/Stock_Close_Predictions.csv")
Dash_df= pd.DataFrame(input_CSV)
Dash_df.set_index("Date", inplace=True, drop=True)
ticker = Dash_df.iloc[0,2]

# Build graph using data from dataframe
fig = go.Figure()
fig.add_trace(go.Scatter(x=Dash_df.index, y=Dash_df.iloc[:,0],
                    mode='lines',
                    name='Actual Close Price ($ USD)'))
fig.add_trace(go.Scatter(x=Dash_df.index, y=Dash_df.iloc[:,1],
                    mode='lines',
                    name='Predicted Close Price ($ USD)'))  
fig.show()

# Set up html elements
app.layout = html.Div(children=[
  html.H1(children = (f'{ticker} Stock Prediction')),

  html.Div(children = '''
    Cyber-Booleans: For All Mankind
    '''),
    # Plot data on graph
    dcc.Graph(
      id='graph',
      figure = fig
    
    )
])

# Run Dash app
if __name__ == '__main__':
    app.run_server(debug=False)