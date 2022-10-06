import pandas as pd
import pandas_datareader.data as web


def load_data(tickers, data_source='fred'):
    return web.DataReader(tickers, data_source).dropna(how='all').ffill()


def clean_data(data, period):
    data['FDHBFIN'] = data['FDHBFIN'] * 1000
    data['GOV_PCT'] = data['TREAST'] / data['GFDEBTN']
    data['HOM_PCT'] = data['FYGFDPUN'] / data['GFDEBTN']
    data['FOR_PCT'] = data['FDHBFIN'] / data['GFDEBTN']
    columns = [
        'DGS1MO', 'DGS3MO', 'DGS1', 'DGS2', 
        'DGS5', 'DGS7', 'DGS10', 'DGS30', 
        'GOV_PCT', 'HOM_PCT', 'FOR_PCT', 'BAA10Y'
    ]
    
    y = data.loc[:, ['DGS1MO', 'DGS5', 'DGS30']].shift(-period)
    y.columns = [col + '_pred' for col in y.columns]
    x = data.loc[:, columns]
    dataset = pd.concat([x, y], axis=1).dropna().iloc[::period, :]
    X = dataset.loc[:, x.columns]
    Y = dataset.loc[:, y.columns]

    return (X, Y)
