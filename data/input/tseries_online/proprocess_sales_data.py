import numpy as np
import pandas
import pickle
import gzip
import datetime

import pandas as pd, numpy as np, random, copy, os, sys
from sklearn.model_selection import train_test_split
random.seed(100)

root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
print(root)

quantitative_columns = ["Quan_4", "Quan_5", "Quan_6", "Quan_7", "Quan_8", "Quan_9", "Quan_10", "Quan_11", "Quan_12", "Quan_13", "Quan_14", "Quan_15", "Quan_16", "Quan_17", "Quan_18", "Quan_19", "Quan_21", "Quan_22", "Quan_27", "Quan_28", "Quan_29", "Quant_22", "Quant_24", "Quant_25"]
def preprocess(dataframe_train, dataframe_test):
    """
    fill null values with median of the column
    Dates are inserted multiple times: number of days, year, month, day
    and binary vectors for year, month and day.
    finally redundant columns are removed
    """
    dataframe = pd.concat([dataframe_train, dataframe_test])
    dataframe['Date_3'] = dataframe.Date_1 - dataframe.Date_2
    train_size = dataframe_train.shape[0]
    X_categorical = []
    X_quantitative = []
    X_date = []
    X_id = []
    ys = np.zeros((train_size,12), dtype=np.int)
    columns = []
    for col in dataframe.columns:
        if col.startswith('Cat_'):
            columns.append(col)
            uni = np.unique(dataframe[col])
            if len(uni) > 1:
                # binarize categorical variables
                X_categorical.append(uni==dataframe[col].values[:,None])
        elif col.startswith('Quan_') or col.startswith('Quant_'):
            columns.append(col)
            if col in quantitative_columns:
                dataframe[col] = np.log(dataframe[col])
            # if the column is not just full of NaN:
            if (pd.isnull(dataframe[col])).sum() > 1:
                tmp = dataframe[col].copy()
                # filling missing values with median
                tmp = tmp.fillna(tmp.median())
                X_quantitative.append(tmp.values)
        elif col.startswith('Date_'):
            columns.append(col)
            # if the column is not just full of NaN
            tmp = dataframe[col].copy()
            if (pd.isnull(tmp)).sum() > 1:
                # median imputation:
                tmp = tmp.fillna(tmp.median())
            X_date.append(tmp.values[:,None])
            # extract day/month/year for seasonal info
            year = np.zeros((tmp.size,1))
            month = np.zeros((tmp.size,1))
            day = np.zeros((tmp.size,1))
            for i, date_number in enumerate(tmp):
                date = datetime.date.fromordinal(int(date_number))
                year[i,0] = date.year
                month[i,0] = date.month
                day[i,0] = date.day
            X_date.append(year)
            X_date.append(month)
            X_date.append(day)
            # consider year, month day as categorical
            X_date.append((np.unique(year)==year).astype(np.int))
            X_date.append((np.unique(month)==month).astype(np.int))
            X_date.append((np.unique(day)==day).astype(np.int))
        elif col=='id':
            pass # X_id.append(dataframe[col].values)
        elif col.startswith('Outcome_'):
            outcome_col_number = int(col.split('M')[1]) - 1
            tmp = dataframe[col][:train_size].copy()
            # median imputation:
            tmp = tmp.fillna(tmp.median())
            ys[:,outcome_col_number] = tmp.values
        else:
            raise NameError

    X_categorical = np.hstack(X_categorical).astype(np.float)
    X_quantitative = np.vstack(X_quantitative).astype(np.float).T
    X_date = np.hstack(X_date).astype(np.float)

    X = np.hstack([X_categorical, X_quantitative, X_date])
    X_train = X[:train_size,:]
    X_test = X[train_size:,:]
    return X_train, X_test, ys, columns

def save_data(data_info, base_path):
    data_info['train_features'].to_csv(f"{base_path}/train/train_features.csv.zip",compression = 'gzip')
    data_info['test_features'].to_csv(f"{base_path}/test/test_features.csv.zip",compression = 'gzip')
    data_info['targets'].to_csv(f"{base_path}/train/targets.csv.zip",compression = 'gzip')

if __name__ == '__main__':

    train_path = 'raw/TrainingDataset.csv'
    test_path = 'raw/TestDataset.csv'
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)


    print ("dataframe_train:", df_train)
    print ("dataframe_test:", df_test)
    
    ids = df_test.values[:,0].astype(np.int)

    X_train, X_test, targets, columns = preprocess(df_train, df_test)
    X_train = pd.DataFrame(X_train)
    X = np.vstack([X_train, X_test])
    X_train = pd.DataFrame( X[:X_train.shape[0], :])
    X_test = pd.DataFrame(X[X_train.shape[0]:, :])
    targets = pd.DataFrame(targets)
    print ("Saving dataset.")
    dataset_info = {"train_features": X_train,
                "test_features": X_test,
                "features_columns": columns,
                "targets": targets}
    save_data(dataset_info,root)
