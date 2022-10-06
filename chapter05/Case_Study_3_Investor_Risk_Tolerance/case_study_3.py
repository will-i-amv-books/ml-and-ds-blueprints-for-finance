import os
import numpy as np
import pandas as pd


def load_data(filename='SCFP2009panel'):
    if os.path.exists(filename + '.csv'):
        dataset = pd.read_csv(filename + '.csv')
    else:
        # Convert dataset to CSV for faster loading
        dataset = pd.read_excel(filename + '.xlsx')
        dataset.to_csv(filename + '.csv')
    return dataset


def clean_data(dataset):
    # Average SP500 during 2007 and 2009
    Average_SP500_2007 = 1478
    Average_SP500_2009 = 948

    # Risk Tolerance 2007
    dataset['RiskFree07'] = dataset['LIQ07'] + dataset['CDS07'] + dataset['SAVBND07'] + dataset['CASHLI07']
    dataset['Risky07'] = dataset['NMMF07'] + dataset['STOCKS07'] + dataset['BOND07'] 
    dataset['RT07'] = dataset['Risky07'] / (dataset['Risky07'] + dataset['RiskFree07'])

    # Risk Tolerance 2009
    dataset['RiskFree09']= dataset['LIQ09'] + dataset['CDS09'] + dataset['SAVBND09'] + dataset['CASHLI09']
    dataset['Risky09'] = dataset['NMMF09'] + dataset['STOCKS09'] + dataset['BOND09'] 
    dataset['RT09'] = (dataset['Risky09'] / (dataset['Risky09'] + dataset['RiskFree09'])) * (Average_SP500_2009 / Average_SP500_2007)

    # Percentage change in risk tolerance between 2007 and 2009. 
    dataset['PercentageChange'] = np.abs((dataset['RT09'] / dataset['RT07']) - 1)

    # Drop the rows containing NaN
    dataset = dataset.dropna(axis=0)
    dataset = dataset[~dataset.isin([np.nan, np.inf, -np.inf]).any(1)]
    
    return dataset


def calc_risk_tolerance(dataset):
    # We consider the intelligent investors whose risk tolerance change 
    # between 2007 and 2009 was less than 10%
    dataset = dataset[dataset['PercentageChange'] <= 0.1]
    
    # We consider the true risk tolerance as the average risk tolerance 
    # of intelligent investors between 2007 and 2009. 
    dataset['TrueRiskTolerance'] = (dataset['RT07'] + dataset['RT09']) / 2

    # Drop labels which might not be needed for the prediction. 
    dataset.drop(labels=['RT07', 'RT09'], axis=1, inplace=True)
    dataset.drop(labels=['PercentageChange'], axis=1, inplace=True)

    return dataset