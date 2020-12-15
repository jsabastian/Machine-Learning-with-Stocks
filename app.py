# !pip install alpha-vantage
# !pip install python-dotenv
from alpha_vantage.timeseries import TimeSeries
# Import API key
# from dotenv import load_dotenv
import os
import plotly.express as px
# Add variable from environments
# load_dotenv()
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import app
import scipy
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import datetime as dt
import pandas as pd
import pandas_market_calendars as mcal

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

# img = ('images/Figure_1.png')

# def Predict_Stock_Prices():
  # Get stock ticker input from user
ticker = 'AAPL'
api_key =  'D46872A04A9M1143LCC'
# Pull and update fields from Alpha Vantage
# try:
ts = TimeSeries(key=api_key, output_format='pandas', indexing_type='integer')
data, meta_data = ts.get_daily_adjusted(symbol=ticker, outputsize='full')
data = data.drop(columns=['7. dividend amount','8. split coefficient'])
df = data.rename(columns={'index':'date','1. open':'open', '2. high':'high', '3. low':'low',
'4. close':'close','5. adjusted close':'adj close','6. volume':'volume'})
# Arrange dataframe in sequential index/date order
df = df[::-1]
df.reset_index(drop=True, inplace=True)
df.head()
df.insert(0,"index", df.index)
data_set = df
data_set.head()

# Pull close data from dataframe and reshape
close_data_set = data_set["close"].values.reshape(-1,1)
dataLength = len(close_data_set)
# Scale to normalize dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data_set = scaler.fit_transform(close_data_set)
# import math as m
# # Split into train and test sets sequentially for LSTM model
trainSize = int(dataLength * 0.8)
testSize = int(dataLength) - trainSize
train_data, test_data = scaled_data_set[0:trainSize,:], scaled_data_set[trainSize:dataLength,:]

xTrain = []
yTrain = []
# Amount of prior days to take into account for each day's prediction
lookbackWindow = 365
# Loop through data and assign lookback data to "prediction" actual value
for i in range(lookbackWindow, trainSize):
    xTrain.append(train_data[i-lookbackWindow:i, 0])
    yTrain.append(train_data[i, 0])
# Reshape lists into arrays for input into LSTM model 
xTrain, yTrain = np.array(xTrain), np.array(yTrain)
xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))

# Model starts as sequential and is assigned LSTM, Dropout, and Dense layers (See readme for parameter details)
model = Sequential()
model.add(LSTM(50, return_sequences = True, input_shape = (xTrain.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))
# Compile and fit model with train data to be used for predictions
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(xTrain, yTrain, epochs = 1, batch_size = 32, verbose = 1, validation_split=0.2)
# model.summary()
# model.save("LSTM_Base_Model.h5")
xTest = []
# Loop through data and assign lookback data to "prediction" true value
for i in range(lookbackWindow, testSize):
    xTest.append(test_data[i-lookbackWindow:i, 0])
# Reshape lists into arrays for input into LSTM model     
xTest = np.array(xTest) 
xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))
# Run model to predict stock close price and scale to actual dollar values
test_data_predictions = model.predict(xTest)
final_test_predictions = scaler.inverse_transform(test_data_predictions)

# Pull last lookbackWindow from test dataset to make first prediction in the future
live_data=test_data[-lookbackWindow:].reshape(1,-1)
rolling_data=live_data[0].tolist()
future_predictions=[]
# Number of days being predicted into the future
predictWindow=60
# Predict first future stock close price and add to both lists
live_data = live_data.reshape((1, lookbackWindow,1))
newDayClose = model.predict(live_data)
# Add new predicted value to lists
rolling_data.extend(newDayClose[0].tolist())
future_predictions.extend(newDayClose.tolist())
# Loop continues to append future predictions to the list to predict the next future stock price
i=1
while(i<predictWindow):
  # Shift window one to the right and reshape  
  live_data = np.array(rolling_data[1:])
  live_data = live_data.reshape(1,-1)
  live_data = live_data.reshape((1, lookbackWindow, 1))
  # Predict next stock price using new window
  newDayClose = model.predict(live_data)
  # Add new predicted value to lists
  rolling_data.extend(newDayClose[0].tolist())
  future_predictions.append(newDayClose)
  # Drop first value in list after making prediction
  rolling_data.pop(0)
  i=i+1
# Scale to actual dollar values
final_future_predictions = scaler.inverse_transform(future_predictions)
# !pip install pandas-market-calendars

# import matplotlib
# # matplotlib.use("TkAgg")
# from matplotlib import pyplot as plt
# plt.style.use("seaborn")
# Connect to market-calendar (only includes dates that the market is open)
nyse = mcal.get_calendar('NYSE')
# Pull dates starting at earliest date for ticker, and end in the future based on how predictWindow
schedule = nyse.schedule(start_date=df["date"].iloc[trainSize+lookbackWindow], end_date=dt.date.today()+ dt.timedelta(days=predictWindow*1.2))
schedule["market_close"] = pd.to_datetime(schedule["market_close"]).dt.date
NYSE_Dates = schedule["market_close"].astype(str).to_list()
# Increment dates every 20 to see on X axis
Clean_NYSE_Dates = NYSE_Dates[0::20]
print(df["date"].iloc[trainSize])
print(Clean_NYSE_Dates)
print(schedule)
# Combine test predictions with future predictions to compare to actual test data
predictPlot = pd.DataFrame(list(final_test_predictions) + list(final_future_predictions))
  # Establish Plot
# plt.figure(figsize=(40,15))
# plt.title(f'Stock Price model for {ticker}')
# plt.xlabel('Date', fontsize=20)
# plt.ylabel('Stock Price at Close (US Dollar)', fontsize=20)
# # Increment dates by same factor as 'skips' in Dates list to plot accurate datestamps on X axis
# plt.xticks(range(0,dataLength,20),Clean_NYSE_Dates, rotation=45)
# # Plot actual values and predictions on same graph
# plt.plot(close_data_set[trainSize+lookbackWindow:])
# plt.plot(predictPlot)
# plt.legend(["Actual Stock Prices", "Predicted Stock Prices"])
  # plt.show()
  # except:
  #   print(f"Error trying to import {ticker}")
act_close = pd.DataFrame(close_data_set)

# Predict_Stock_Prices()

fig = px.scatter(predictPlot)
fig_actual = px.scatter(act_close)

app.layout = html.Div(children=[
  html.H1(children = (f'{ticker} Stock Prediction')),

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