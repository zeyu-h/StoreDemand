import numpy as np
import pandas as pd
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

BOOSTER = 'gbdt'
FEATURE_SCORE_FILE = 'feature_score_%s.csv'%BOOSTER
NULL_IMPORTANCE_FILE = 'null_importances_distribution_%s.csv'%BOOSTER
REAL_IMPORTANCE_FILE = 'actual_importances_ditribution_%s.csv'%BOOSTER

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# Symmetric mean absolute percentage error
def smape_score(y_true, y_pred):
    #return np.mean( np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)) ) * 100
    return np.mean(np.abs(y_true - y_pred) / max(y_true, + 1)) * 100


def calculate_score_df(actual_imp_df, null_imp_df):
    feature_scores = []
    for _f in actual_imp_df['feature'].unique():
        f_null_imps_gain = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
        f_act_imps_gain = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].mean()
        gain_score = np.log(1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75)))

        f_null_imps_split = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
        f_act_imps_split = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].mean()
        split_score = np.log(1e-10 + f_act_imps_split / (1 + np.percentile(f_null_imps_split, 75)))
        feature_scores.append((_f, split_score, gain_score))

    scores_df = pd.DataFrame(feature_scores, columns=['feature', 'split_score', 'gain_score'])
    scores_df.to_csv(FEATURE_SCORE_FILE)
    return scores_df


def calculate_correlation_score_df(actual_imp_df, null_imp_df):
    correlation_scores = []
    for _f in actual_imp_df['feature'].unique():
        f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
        f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].values
        gain_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
        f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
        f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].values
        split_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
        correlation_scores.append((_f, split_score, gain_score))

    corr_scores_df = pd.DataFrame(correlation_scores, columns=['feature', 'split_score', 'gain_score'])
    return corr_scores_df


def get_feature_importances(data, real_or_null, frac=0.5):
    # Gather real features
    train_features = [f for f in data.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    data = data[data['TARGET'].notnull()].sample(frac=frac)

    X = data[train_features]
    y = data['TARGET'] if real_or_null.upper()=='REAL' else data['TARGET'].sample(frac=1.0)

    # Go over fold and keep track of CV score (train and valid) and feature importances
    train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.25)

    clf = LGBMClassifier(
        nthread=7,
        objective='regression_l1',
        n_estimators=10000,
        learning_rate=0.1,
        num_leaves=98,
        colsample_bytree=0.3489842602836827,
        subsample=0.8715623,
        max_depth=6,
        reg_alpha=3,
        reg_lambda=2.,
        min_split_gain=0.004,
        min_child_weight=6.9,
        min_child_samples=200,
        silent=-1,
        verbose=-1)

    # Fit the model
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
            eval_metric='smape', verbose=200, early_stopping_rounds=200)

    # Get feature importances
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(train_features)
    imp_df["importance_gain"] = clf.booster_.feature_importance(importance_type='gain')
    imp_df["importance_split"] = clf.booster_.feature_importance(importance_type='split')
    imp_df['trn_score'] = smape_score(valid_y, clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:,1])

    return imp_df