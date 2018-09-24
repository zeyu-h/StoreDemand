import numpy as np
import pandas as pd
import time
from contextlib import contextmanager

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

magic_number = 1.00

def store_item_factor(df):
    store_item_table = pd.pivot_table(df, index='store', columns='item',
                                  values='sales', aggfunc=np.mean)
    return store_item_table


def month_effect(df, grand_avg):
    month_table = pd.pivot_table(df, index='month', values='sales', aggfunc=np.mean)
    month_table.sales /= grand_avg
    return month_table


    # Day of week pattern
def weekday_effect(df, grand_avg):
    weekday_table = pd.pivot_table(df, index='weekday', values='sales', aggfunc=np.mean)
    weekday_table.sales /= grand_avg
    return weekday_table


def year_effect(df, grand_avg, decay_factor):
    year_table = pd.pivot_table(df, index='year', values='sales', aggfunc=np.mean)
    year_table /= grand_avg

    years = np.arange(2013, 2019)
    annual_sales_avg = year_table.values.squeeze()

    if decay_factor:
        weights = np.exp((years - 2018) / decay_factor)
        annual_growth = np.poly1d(np.polyfit(years[:-1], annual_sales_avg, 2, w=weights[:-1]))
    else:
        #p1 = np.poly1d(np.polyfit(years[:-1], annual_sales_avg, 1))
        annual_growth = np.poly1d(np.polyfit(years[:-1], annual_sales_avg, 2))
    print(f"2018 Relative Sales by Weighted Fit = {annual_growth(2018)}")
    return annual_growth



def improved_factor(df, cut_off_year):
    cut_off_df = df.loc[df.year>=cut_off_year]
    grand_avg = cut_off_df.sales.mean()

    # Day of week - Item Look up table
    weekday_item_table = pd.pivot_table(cut_off_df, index='weekday', columns='item', values='sales', aggfunc=np.mean)

    # Month pattern
    month_table = pd.pivot_table(cut_off_df, index='month', values='sales', aggfunc=np.mean)
    month_table.sales /= grand_avg

    # Store pattern
    store_table = pd.pivot_table(cut_off_df, index='store', values='sales', aggfunc=np.mean)
    store_table.sales /= grand_avg
    return grand_avg, weekday_item_table, month_table, store_table



def factor_predictor(test, submission, store_item_table, month_table, weekday_table, annual_growth, round=True):
    submission[['sales']] = submission[['sales']].astype(np.float64)
    for _, row in test.iterrows():
        dow, month, year = row.weekday, row.month, row.year
        item, store = row['item'], row['store']
        base_sales = store_item_table.at[store, item]
        mul = month_table.at[month, 'sales'] * weekday_table.at[dow, 'sales']
        pred_sales = base_sales * mul * annual_growth(year) * magic_number
        submission.at[row['id'], 'sales'] = pred_sales

    if round:
        submission['sales'] = np.round(submission['sales']).astype(int)
    submission.to_csv("factor_model_v1.csv", index=False)
    return



def awesome_predictor(test, submission, weekday_item_table, month_table, store_table, annual_growth, round=True):
    submission[['sales']] = submission[['sales']].astype(np.float64)
    for _, row in test.iterrows():
        dow, month, year = row.weekday, row.month, row.year
        item, store = row['item'], row['store']
        base_sales = weekday_item_table.at[dow, item]
        mul = month_table.at[month, 'sales'] * store_table.at[store, 'sales']
        pred_sales = base_sales * mul * annual_growth(year) * magic_number
        submission.at[row['id'], 'sales'] = pred_sales
    if round:
        submission['sales'] = np.round(submission['sales']).astype(int)
    submission.to_csv("factor_model_v2.csv", index=False)
    return submission



def sales_train_test():
    # Read data and merge
    train_df = pd.read_csv('../input/train.csv')
    test_df = pd.read_csv('../input/test.csv')

    def expand_df(df):
        df['date'] = pd.to_datetime(df['date'])
        df['weekday'] = df['date'].dt.weekday
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['year'] = df['year']
        #df = df[df['month']<6]
        return df
    return expand_df(train_df), expand_df(test_df)


factor_method = 'v2'
def main():
    train_df, test_df = sales_train_test()
    test_df['sales'] = 0.
    submission = test_df[['id', 'sales']].copy()

    if factor_method == 'v1':
        global_sales_mean = np.mean(train_df['sales'])
        store_item_table = store_item_factor(train_df)
        weekday_table = weekday_effect(train_df, global_sales_mean)
        month_table = month_effect(train_df, global_sales_mean)
        annual_growth = year_effect(train_df, global_sales_mean, decay_factor=5)

        factor_predictor(test_df, submission, store_item_table, month_table, weekday_table, annual_growth)
    elif factor_method == 'v2':
        global_sales_mean, weekday_item_table, month_table, store_table = improved_factor(train_df, 2015)
        annual_growth = year_effect(train_df, global_sales_mean, decay_factor=9)
        awesome_predictor(test_df, submission, weekday_item_table, month_table, store_table, annual_growth)


if __name__ == "__main__":
    with timer("Full model run"):
        main()