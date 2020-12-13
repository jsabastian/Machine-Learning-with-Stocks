// test structure for me to remember format
  // going to start trying to replicate our code to pull down ticker data

function GetTickerData() {

  var symbol = prompt('Enter a Ticker Label:');
  var outputsize = 'full';
  var datatype = 'json';
  alpha.data.daily(symbol, outputsize, datatype).then((d) => {
    var results = JSON.stringify(d)
    var test = JSON.parse(results);
    console.log(test);
    var keys = Object.keys(test['Time Series (Daily)']);
    var datalength = Object.keys(test['Time Series (Daily)']).length;
    console.log(datalength);
    var latestkey = keys[0];
    close_values = [];
    dates = [];
    var getData = test['Time Series (Daily)'][latestkey]['4. close'];
    for (i = 0; i < datalength; i++) {
        close_values.push(test['Time Series (Daily)'][(keys[i])]['4. close']);
      };
    for (i = 0; i < datalength; i++) {
    dates.push(Object.keys(test['Time Series (Daily)'])[i]);
    };
    console.log(close_values.length);
    console.log(dates.length)
    return close_values;
  });
};

GetTickerData();