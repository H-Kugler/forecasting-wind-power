from statsmodels.tsa.stattools import adfuller


# it is necessary to check if the data is stationary. We can do this by using the Augmented Dickey-Fuller test
def check_stationarity(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'{key}: {value}')