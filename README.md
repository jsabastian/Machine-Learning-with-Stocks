## Welcome to our Machine-Learning Stock Predictor

### !!DISCLAIMER!!
##### This project is meant for educational purposes ONLY. It is a study of Machine Learning models and not meant to be used for actual stock price prediction. Buy/Sell at your own risk.

#### The machine learning model makes use of the Long Short-Term Memory Recurrent Nerual Network architecture to turn time-series data into a multi-feature array to be used by Keras and Tensorflow, allowing us to run Sequential modeling on the data. At a basic level, each datapoint is trained (and later predicted) by a window of datapoints prior to it. That is to say, tomorrow's stock price is predicted using today's, and yesterday's, etc. This variable in our code is "lookbackWindow" and can be adjusted accordingly - *Our model uses an entire year*

#### The ML_stock_predictions.ipynb file is a stand alone notebook that runs the model and outputs a chart and CSV of the data. *Please note some of the libraries may require pip installs.* Once run, the script will request the user to enter a stock ticker. *(note that some newer stocks may not work if the lookbackWindow is too large compared to the data)*

#### Once a valid stock ticker is entered, the API will automatically pull the entirety of that company's stock data and run the script. For implementation on Heroku, these changes must be pushed to the GitHub repository, which will automatically update the chart online. This happens automatically through app.py, since it pulls the data from a CSV file, which is updated by the ML_stock_predictions.ipynb file. It is done this way due to the limitation of Heroku, which does not have the processing power to run the Machine Learning code on it's own.


### For more in-depth information about our model and the parameters used, please visit our project landing page here: https://coreygaunt.github.io/ml-stock-predictions/

### Link to Heroku Deployment: https://guarded-castle-81488.herokuapp.com/