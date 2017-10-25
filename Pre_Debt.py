import pandas as pd
import numpy as np
import math


def read_file(filename):
    dataframe = pd.read_csv(filename,nrows=1000)
    dataframe.drop('id',1,inplace=True)
    return dataframe

def pre_nan(dataframe):
    dataframe.dropna(how='all', inplace=True)

    df_columns = dataframe.columns
    drop_columns = []
    for col in df_columns:
        percent = round(dataframe[col].count()/len(dataframe.index),2)
        # Column Value exist above 20% data is default value , Drop it.
        if percent < 0.8:
            drop_columns.append(col)
        # Fill Default data use before value.
        dataframe[col].fillna(method='ffill', inplace=True)

        # cast dtype to float64 if can transfer.
        try:
            dataframe[col] = dataframe[col].astype('float64')
        except:
            # some object can't transfer
            dataframe[col] = dataframe[col]
        # Value >= E+10 column ,Drop it.
        if dataframe[col].dtype in ('float64','int64'):
            col_mean,col_median,col_var = dataframe[col].mean(),dataframe[col].median(),dataframe[col].var()
            # if col value all big num or the same num , Drop the column.
            if col_mean> 0:
                if math.floor(math.log10(col_mean)) + 1 >= 10:
                    drop_columns.append(col)

            if col_var == 0:
                drop_columns.append(col)

    dataframe.drop(drop_columns,axis = 1,inplace=True)
    return dataframe

def pre_pearson(dataframe):
    nrow, ncol = np.shape(dataframe)
    # df_columns = dataframe.columns
    # dict_corr_cols = {}
    corr_cols = []
    for i in range(ncol):

        # calc correspondence matrix
        corr = dataframe.iloc[:, i + 1:].corrwith(dataframe.iloc[:, i])
        # filter higher pairs
        filter_coll = corr[abs(corr) >= 0.9]
        if np.shape(filter_coll)[0] > 0:
            # seri = [[ser[0],ser[1]] for ser in zip(filter_coll.index,filter_coll.values)]
            # current_col = df_columns[i]
            # dict_corr_cols[current_col] = seri
            # print (dict_corr_cols)
            corr_cols += filter_coll.index.tolist()

    dataframe.drop(corr_cols, axis=1, inplace=True)
    return dataframe

def pre_discretization(dataframe):
    df_columns = dataframe.columns
    nrow,ncol = np.shape(dataframe)
    for col in df_columns:
        vc = dataframe[col].value_counts()

        large_vc = vc[vc>=0.9*nrow]
        if np.shape(large_vc)[0] > 0 :
            dataframe.drop(col, axis=1, inplace=True)
    for col in dataframe.columns:
        print (col)


def Random_Forest(dataframe):
    nrow, ncol = np.shape(dataframe)

if __name__ == '__main__':
    filename = r'C:\Users\IBM_ADMIN\Downloads\train_v2.csv\train_v2.csv'
    df = read_file(filename)
    df = pre_pearson(pre_nan(df))
    df = pre_discretization(df)