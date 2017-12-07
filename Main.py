import PreProcess as pre
import xgboost as xgb
import pandas as pd
import matplotlib as plt

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

# Read ordinal features meta data
ordinal_arrays = {}
with open("./meta_data.csv", mode="r") as f:
    first = True
    for line in f:
        line = line[:-1]
        splited = line.split(',')
        ordinal_arrays[splited[0]] = splited[1:]

for column_name, column_type in train.dtypes.iteritems():
    if column_name == "SalePrice":
        continue
    if column_name == "MSSubClass":
        train, test = pre.process_categorical(train, test, column_name)
    elif column_name in ordinal_arrays.keys():
        train, test = pre.process_ordinal(train, test, column_name, ordinal_arrays[column_name])
    elif column_type != object:
        train, test = pre.process_numeric(train, test, column_name)
    else:
        train, test = pre.process_categorical(train, test, column_name)

# contains all the columns names ( without 'SalePrice' and without 'Id' )
columns = list(test)
columns.remove("Id")

# Convert the pandas DF to 2D numpy array
train_matrix = train.as_matrix(columns)
test_matrix = test.as_matrix(columns)
label_matrix=train["SalePrice"].values

xgb_params={'booster': 'dart','eta': 0.3, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8,
             'max_depth':8, 'min_child_weight':4,
            'rate_drop': 0.1,'skip_drop': 0.5,'normalize_type': 'tree'}

def modelfit(xgb_params, train_matrix, test_matrix, label_matrix, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    #tune number of trees using cv
    xgtrain = xgb.DMatrix(train_matrix, label=label_matrix)
    cvresult = xgb.cv(xgb_params, xgtrain, num_boost_round=10000, nfold=cv_folds,
                      metrics='rmse', early_stopping_rounds=early_stopping_rounds)

    #train the model
    model=xgb.train(xgb_params,xgtrain,cvresult.shape[0])

    #test to DMatrix format
    xgTest=xgb.DMatrix(test_matrix)

    #get prediction
    res=model.predict(xgTest)
    print(res)


    ids=range(1461,(1461+len(res)))
    result_df=pd.DataFrame({"Id":ids,"SalePrice":res})
    result_df.to_csv("./submission2.csv")
    # # Predict training set:
    # dtrain_predictions = alg.predict(dtrain[predictors])
    # dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]
    #
    # # Print model report:
    # print ("\nModel Report")
    # print ("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))
    # print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))
    #
    # feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    # plt=feat_imp.plot(kind='bar', title='Feature Importances')
    # plt.ylabel('Feature Importance Score')

modelfit(xgb_params,train_matrix, test_matrix,label_matrix)