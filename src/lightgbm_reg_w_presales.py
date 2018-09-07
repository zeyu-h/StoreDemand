import numpy as np
import pandas as pd
import time
from contextlib import contextmanager
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import gc
import copy

SEED = 987

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

holiday = {
    'Washington Day':['2013-02-18', '2014-02-17', '2015-02-16', '2016-02-15', '2017-02-20', '2018-02-19'],
    'Martin Luther King Day': ['2013-01-21', '2014-01-20', '2015-01-19', '2016-01-18', '2017-01-16', '2018-01-15'],
    'New Years Day':['2013-01-01', '2014-01-01', '2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01'],
    'New Years Day After':['2013-01-02', '2014-01-02', '2015-01-02', '2016-01-02', '2017-01-02', '2018-01-02']
}

# Symmetric mean absolute percentage error
def smape_func(y_true, y_pred):
    #return np.mean( np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)) ) * 100
    return np.mean(np.abs(y_true - y_pred) / np.maximum(y_true, + 1)) * 100

kernels = {
    # 'gbdt':{'colsample_bytree': 0.4683, 'max_bin': 127, 'min_child_samples': 2000, 'min_child_weight': 0.001305,
    #    'min_split_gain': 0.000696, 'num_leaves': 15, 'reg_alpha': 6.430, 'reg_lambda': 0.005522, 'subsample': 0.8477}
    'gbdt': {
        'colsample_bytree': 0.9, 'max_bin': 127, 'min_child_samples': 20, 'min_child_weight': 6.996,
        'min_split_gain': 0.0373, 'num_leaves': 64, 'max_depth': 5,
        'reg_alpha': 3.1, 'reg_lambda': 2.9,
        'subsample': 0.8477, 'subsample_freq':5
    }
}

def trend_predict(model, test_df, feats):
    results = {}
    for year in [2013, 2014, 2015, 2016, 2017]:
        test_df['year'] = year
        results[year] = model.predict(test_df[feats])
    trend_result = results[2017] + \
                   0.4 * (results[2017] - results[2016]) + \
                   0.3 * (results[2016] - results[2015]) + \
                   0.2 * (results[2015] - results[2014]) + \
                   0.1 * (results[2014] - results[2013])
    return trend_result

def kfold_lightgbm(df, num_folds, booster, num_rows=None):
    # Divide in training/validation and test data
    train_df = copy.deepcopy(df[df['sales'].notnull()])
    test_df = copy.deepcopy(df[df['sales'].isnull()])
    if num_rows:
        train_df, test_df = train_df.iloc[:num_rows], test_df.iloc[:num_rows]
    test_df['id'] = test_df['id'].astype(int)
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    folds = KFold(n_splits=num_folds, shuffle=True, random_state=SEED)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feats = [f for f in train_df.columns if f not in ['id', 'sales', 'index']]

    smape_score = np.zeros(num_folds, float)

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['sales'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['sales'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['sales'].iloc[valid_idx]

        params = kernels[booster]
        lgb_param = {
            'nthread': 7,
            'boosting_type': booster,
            'metric': 'mape',
            'objective': 'regression_l1',
            #'max_bin': params['max_bin'],
            'learning_rate': 0.2,
            'num_leaves': params['num_leaves'],
            'colsample_bytree': params['colsample_bytree'],
            'subsample': params['subsample'],
            'max_depth': params['max_depth'],
            'subsample_freq': params['subsample_freq'],
            'reg_alpha': params['reg_alpha'],
            'reg_lambda': params['reg_lambda'],
            'min_split_gain': params['min_split_gain'],
            'min_child_weight': params['min_child_weight'],
            'min_child_samples': params['min_child_samples'],
            'verbose':-1
        }

        cat_col = feats  #['holiday', 'year', 'month', 'week', 'weekday', 'store']
        lgb_train = lgb.Dataset(data=train_x, label=train_y, categorical_feature=cat_col)
        lgb_valid = lgb.Dataset(data=valid_x, label=valid_y, categorical_feature=cat_col)

        model = lgb.train(params=lgb_param, train_set=lgb_train, num_boost_round=5000, valid_sets=[lgb_train, lgb_valid],
                              verbose_eval =100, early_stopping_rounds=200)

        oof_preds[valid_idx] = model.predict(valid_x)
        #sub_preds += model.predict(test_df[feats]) / folds.n_splits
        sub_preds += trend_predict(model, test_df, feats) / folds.n_splits

        smape_score[n_fold] = smape_func(valid_y, oof_preds[valid_idx])
        print('Fold %2d SMAPE : %.3f' % (n_fold + 1, smape_score[n_fold]))
        del train_x, train_y, valid_x, valid_y

    full_smape = smape_func(train_df['sales'], oof_preds)
    print('Full SMAPE score %.6f' % full_smape)
    print('SMAPE scores mean = %.6f, std = %.6f'%(smape_score.mean(), smape_score.std()))
    # Write submission file and plot feature importance
    sub_df = test_df[['id']].copy()
    sub_df['sales'] = sub_preds
    submission_file_name = 'lightgbm_%s_%.3f.csv' % (booster, full_smape)
    sub_df[['id', 'sales']].to_csv(submission_file_name, index= False)
    return

def sales_train_test(num_rows=None):
    # Read data and merge
    df = pd.read_csv('../input/train.csv', nrows=num_rows)
    test_df = pd.read_csv('../input/test.csv', nrows=num_rows)

    df = df.append(test_df, sort=False).reset_index()

    #put holiday
    coder = 1
    df['holiday'] = 0
    for key in holiday:
        df.loc[df['date'].isin(holiday[key]), 'holiday'] = coder
        coder += 1

    # split date to year, month,
    df['date'] = pd.to_datetime(df['date'])
    df['weekday'] = df['date'].dt.weekday
    df['week'] = df['date'].apply(lambda x,:x.isocalendar()[1])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    df['store_item'] = df['store'].apply(lambda x:'%d-'%x)+df['item'].apply(lambda x:'%d'%x)

    df.sort_values(['store_item', 'date'], inplace=True)

    df['week-1_sales'] = df['sales'].shift(7)
    df['week-2_sales'] = df['sales'].shift(14)
    df['day-1_sales'] = df['sales'].shift(1)
    df['day-1_sales'] = df['sales'].shift(2)

    df.drop('date', axis=1, inplace=True)

    return df


def main(debug=False):
    num_rows = 10000 if debug else None
    df = sales_train_test(num_rows)
    kfold_lightgbm(df, num_folds=5, booster='gbdt', num_rows=num_rows)

if __name__ == "__main__":
    with timer("Full model run"):
        main(debug=False)
