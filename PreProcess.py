import learn as learn
import pandas as pd
import numpy as np
import sklearn.preprocessing as pre
import sys
import re


def process_numeric(train, test, column_name):
    # get mean of training data
    mean = train[column_name].mean()

    # replace missing data with training set mean
    train[column_name] = train[column_name].fillna(mean)
    test[column_name] = test[column_name].fillna(mean)

    # subtract mean from column for both train and test
    train[column_name] = train[column_name] - mean
    test[column_name] = test[column_name] - mean

    # get standard deviation of training data
    std = train[column_name].std()

    # divide column values by std for both training and test data
    train[column_name] = train[column_name] / std
    test[column_name] = test[column_name] / std

    return train, test


def process_ordinal(train, test, column_name, map_array):
    # Creates the map for the ordinal feature
    map = {}
    for i in range(0, len(map_array)):
        map[map_array[i]] = i

    # Replace by the map
    train[column_name] = train[column_name].fillna(0)
    train[column_name].replace(map, inplace=True)

    test[column_name] = test[column_name].fillna(0)
    test[column_name].replace(map, inplace=True)

    return train, test


def process_categorical(train, test, column_name):
    if column_name == "MSSubClass":
        train["MSSubClass"] = train["MSSubClass"].astype(str)
        test["MSSubClass"] = test["MSSubClass"].astype(str)

    # replace NA with a dummy variable
    train[column_name] = train[column_name].fillna('_missing')
    test[column_name] = test[column_name].fillna('_missing')

    # extract categories for column labels
    # note that .unique() extracts the labels as a numpy array
    labels_train = train[column_name].unique()
    labels_train.sort(axis=0)
    labels_test = test[column_name].unique()
    labels_test.sort(axis=0)

    # transform text classifications to numerical id
    encoder = pre.LabelEncoder()
    cat_train = train[column_name]
    cat_train_encoded = encoder.fit_transform(cat_train)

    cat_test = test[column_name]
    cat_test_encoded = encoder.fit_transform(cat_test)

    # apply onehotencoding
    onehotencoder = pre.OneHotEncoder()
    cat_train_1hot = onehotencoder.fit_transform(cat_train_encoded.reshape(-1, 1))
    cat_test_1hot = onehotencoder.fit_transform(cat_test_encoded.reshape(-1, 1))

    # append column header name to each category listing
    # note the iteration is over a numpy array hence the [...] approach
    labels_train[...] = column_name + '_' + labels_train[...]
    labels_test[...] = column_name + '_' + labels_test[...]

    # convert sparse array to pandas dataframe with column labels
    df_train_cat = pd.DataFrame(cat_train_1hot.toarray(), columns=labels_train)
    df_test_cat = pd.DataFrame(cat_test_1hot.toarray(), columns=labels_test)

    # Get missing columns in test set that are present in training set
    missing_cols = set(df_train_cat.columns) - set(df_test_cat.columns)
    # Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        df_test_cat[c] = 0
    # Ensure the order of column in the test set is in the same order than in train set
    # Note this also removes categories in test set that aren't present in training set
    df_test_cat = df_test_cat[df_train_cat.columns]

    # concatenate the sparse set with the rest of our training data
    train = pd.concat([df_train_cat, train], axis=1)
    test = pd.concat([df_test_cat, test], axis=1)

    # delete original column from training data
    del train[column_name]
    del test[column_name]

    return train, test


def remove_lowDeviation(train, predictors,treshold=0.05):
    res=[]
    for column in predictors:
        if np.std(train[column])<=treshold:
            continue
        res.append(column)
    return res


def remove_lowImportance(train, predictors,treshold=0.045):
    res=[]
    for column in predictors:
        column_target = train[[column,'SalePrice']]
        p=column_target.corr('pearson')["SalePrice"][0]
        k=column_target.corr('kendall')["SalePrice"][0]
        s=column_target.corr('spearman')["SalePrice"][0]
        cor=(p+k+s)/3
        if -1*treshold<cor and cor<treshold:
            continue
        res.append(column)
    return res


def remove_similar(train, predictors, treshold=0.75):
    to_remove = []
    p = train.corr('pearson')
    k = train.corr('kendall')
    s = train.corr('spearman')
    corr_df=p+k+s
    corr_df=corr_df.divide(3)
    overlapping=[]
    for column1 in predictors:
        overlapping.append(column1)
        for column2 in predictors:
            if column2 in overlapping:
                continue
            cor=corr_df.loc[column1,column2]
            if -1 * treshold > cor or cor > treshold:
                if corr_df.loc[column1,"SalePrice"]>corr_df.loc[column2,"SalePrice"]:
                    to_remove.append(column2)
                    overlapping.append(column2)
                else:
                    to_remove.append(column1)
                    break

    return list(set(predictors)-set(to_remove))




def features_selection(train,predictors,tresh_deviation,tresh_importance,tresh_similarity):
    print(len(predictors))
    predictors=remove_lowDeviation(train,predictors,tresh_deviation)
    print(len(predictors))
    predictors=remove_lowImportance(train,predictors, tresh_importance)
    print(len(predictors))
    predictors=remove_similar(train,predictors, tresh_similarity)
    print(len(predictors))


    return predictors