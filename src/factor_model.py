import numpy as np
import pandas as pd
import time
from contextlib import contextmanager
import datetime

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

magic_number = 1.01


def week_number(d):
    return (datetime.date(d.year, d.month, d.day) - datetime.date(d.year, 1, 1)).days // 7 + 1


# predict week day effect, based on week number and item. the trend within a year is caught here
def weekday_effect(df, grand_avg):
    week_weekday_table = df.groupby(['week', 'weekday']).agg({'sales':'mean'})/grand_avg
    return week_weekday_table


def store_item_factor(df):
    store_item_table = pd.pivot_table(df, index='store', columns='item',
                                  values='sales', aggfunc=np.mean)
    return store_item_table


def year_effect(df, grand_avg):
    year_table = pd.pivot_table(df, index='year', values='sales', aggfunc=np.mean)
    year_table /= grand_avg

    years = np.arange(2013, 2019)
    annual_sales_avg = year_table.values.squeeze()

    #p1 = np.poly1d(np.polyfit(years[:-1], annual_sales_avg, 1))
    p2 = np.poly1d(np.polyfit(years[:-1], annual_sales_avg, 2))
    return p2



def slightly_better(test, submission, store_item_table, weekday_table, annual_growth):
    submission[['sales']] = submission[['sales']].astype(np.float64)
    for _, row in test.iterrows():
        dow, week, month, year = row.weekday, row.week, row.month, row.year
        item, store = row['item'], row['store']
        base_sales = store_item_table.at[store, item]
        mul = weekday_table.at[(week, dow), 'sales']
        pred_sales = base_sales * mul * annual_growth(year) * magic_number
        submission.at[row['id'], 'sales'] = pred_sales
    submission.to_csv("factor_wk_wkd.csv", index=False)
    return


def sales_train_test():
    # Read data and merge
    train_df = pd.read_csv('../input/train.csv')
    test_df = pd.read_csv('../input/test.csv')

    def expand_df(df):
        df['date'] = pd.to_datetime(df['date'])
        df['week'] = df['date'].apply(lambda x: week_number(x))
        df['weekday'] = df['date'].dt.weekday
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['year'] = df['year']
        return df
    return expand_df(train_df), expand_df(test_df)


def main():
    train_df, test_df = sales_train_test()
    global_sales_mean = np.mean(train_df['sales'])
    store_item_table = store_item_factor(train_df)
    weekday_table = weekday_effect(train_df, global_sales_mean)
    annual_growth = year_effect(train_df, global_sales_mean)

    test_df['sales'] = 0.
    submission = test_df[['id', 'sales']].copy()
    slightly_better(test_df, submission, store_item_table, weekday_table, annual_growth)

if __name__ == "__main__":
    with timer("Full model run"):
        main()