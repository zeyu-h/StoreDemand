import numpy as np
import pandas as pd
import time
from contextlib import contextmanager

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

magic_number = 1.01

def poly2d(x1, x2, coeff):
    y = coeff[0] + coeff[1]*x1 + coeff[2]*x2 + coeff[3]*x1**2 + coeff[4]*x2**2 + coeff[5]*x1*x2
    return y


def r2_explained(x1, x2, y):
    x1 = x1.flatten()
    x2 = x2.flatten()
    A = np.array([x1*0+1, x1, x2, x1**2, x2**2, x1*x2]).T
    y = y.flatten()
    coeff, r, rank, s = np.linalg.lstsq(A, y)
    r2 = 1 - r / (y.size * y.var())
    return coeff, r2


def month_year_factor(df, grand_avg):
    factors_dict = {}
    min_r2_val = 1.
    min_r2_key = ''
    max_r2_val = 0.
    max_r2_key = ''
    item_store_year_month = df.groupby(['item', 'store', 'year', 'month']).agg({'sales': 'mean'})
    for item in df['item'].unique():
        for store in df['store'].unique():
            x1 = np.array(item_store_year_month.loc[(item, store)].index.get_level_values('year'))
            x2 = np.array(item_store_year_month.loc[(item, store)].index.get_level_values('month'))
            y = np.array(item_store_year_month.loc[(item, store), 'sales'])
            coeff, r2 = r2_explained(x1, x2, y)
            factors_dict[(item, store)] = coeff, r2
            if r2 < min_r2_val:
                min_r2_val = r2
                min_r2_key = 'item %d, store %d, min r2 = %.4f' % (item, store, r2)
            if r2 > max_r2_val:
                max_r2_val = r2
                max_r2_key = 'item %d, store %d, max r2 = %.4f' % (item, store, r2)
    print(min_r2_key)
    print(max_r2_key)
    return factors_dict


def year_month_lookup(item, store, year, month, factor_dict):
    coeff, _ = factor_dict[(item, store)]
    return poly2d(year, month, coeff)


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


def year_effect(df, grand_avg):
    year_table = pd.pivot_table(df, index='year', values='sales', aggfunc=np.mean)
    year_table /= grand_avg

    years = np.arange(2013, 2019)
    annual_sales_avg = year_table.values.squeeze()

    #p1 = np.poly1d(np.polyfit(years[:-1], annual_sales_avg, 1))
    p2 = np.poly1d(np.polyfit(years[:-1], annual_sales_avg, 2))
    return p2



def slightly_better(test, submission, store_item_table, month_table, weekday_table, annual_growth):
    submission[['sales']] = submission[['sales']].astype(np.float64)
    for _, row in test.iterrows():
        dow, month, year = row.weekday, row.month, row.year
        item, store = row['item'], row['store']
        base_sales = store_item_table.at[store, item]
        mul = month_table.at[month, 'sales'] * weekday_table.at[dow, 'sales']
        pred_sales = base_sales * mul * annual_growth(year) * magic_number
        submission.at[row['id'], 'sales'] = pred_sales
    submission.to_csv("sbp_round.csv", index=False)
    return


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


def main():
    train_df, test_df = sales_train_test()
    global_sales_mean = np.mean(train_df['sales'])
    store_item_table = store_item_factor(train_df)
    weekday_table = weekday_effect(train_df, global_sales_mean)
    month_table = month_effect(train_df, global_sales_mean)
    annual_growth = year_effect(train_df, global_sales_mean)

    test_df['sales'] = 0.
    submission = test_df[['id', 'sales']].copy()
    slightly_better(test_df, submission, store_item_table, month_table, weekday_table, annual_growth)

if __name__ == "__main__":
    with timer("Full model run"):
        main()