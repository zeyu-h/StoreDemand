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

magic_number = 1.02

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


def week_number(d):
    return (datetime.date(d.year, d.month, d.day) - datetime.date(d.year, 1, 1)).days // 7 + 1


# predict week day effect, based on week number and item. the trend within a year is caught here
def weekday_effect(df):
    data = df.copy()
    item_year = data.groupby(['item', 'year']).agg({'sales': 'sum', 'date': 'count'})
    data = data.set_index(['item', 'year'])
    data['avg_day_sales'] = item_year['sales']/item_year['date']
    data = data.reset_index()
    data['day_ratio'] = data['sales']/data['avg_day_sales']
    item_week_weekday = data.groupby(['item', 'week', 'weekday']).agg({'day_ratio':'mean'})
    return item_week_weekday


# predict avarage day sales
def year_effect_by_item(df, degree=2):
    item_year = df.groupby(['item', 'year']).agg({'sales': 'mean'})
    year_effect_dict = {}
    for item in df['item'].unique():
        x = np.array(item_year.loc[(item)].index)
        y = np.array(np.array(item_year.loc[item,'sales'].values))*10   #per store per day, *10 to per day
        p = np.poly1d(np.polyfit(x, y, degree))
        year_effect_dict[item] = p
    return year_effect_dict


def store_item_factor(df):
    store_item_sale = df.groupby(['item', 'store']).agg({'sales':'sum'})
    store_item_table = pd.DataFrame(index=store_item_sale.index, columns=['ratio'])
    for item in df['item'].unique():
        store_item_table.loc[item, 'ratio'] = np.array(store_item_sale.loc[item]['sales']/float(np.sum(store_item_sale.loc[item]['sales'])))
    return store_item_table


def sales_train_test():
    # Read data and merge
    train_df = pd.read_csv('../input/train.csv')
    test_df = pd.read_csv('../input/test.csv')

    def expand_df(df):
        df['date'] = pd.to_datetime(df['date'])
        df['week'] = df['date'].apply(lambda x: week_number(x))
        df['weekday'] = df['date'].dt.weekday
        df['year'] = df['date'].dt.year - 2000
        df['month'] = df['date'].dt.month
        df['year'] = df['year']
        #df = df[df['month']<6]
        return df
    return expand_df(train_df), expand_df(test_df)


def sales_predict(row, average_day_sale_dict, item_week_weekday):
    week, weekday, item, store = row['week'], row['weekday'], row['item'], row['store']
    return average_day_sale_dict[(item, store)] * float(item_week_weekday.loc[(item, week, weekday)])


def sale_predict(df, annual_growth_dict, store_item_table, item_week_weekday):
    average_day_sale_dict = {}
    for item in df['item'].unique():
        item_average_day_sale = annual_growth_dict[item](18)
        for store in df['store'].unique():
            average_day_sale_dict[(item, store)] = float(store_item_table.loc[(item, store)])*item_average_day_sale

    df['sales'] = df.apply(lambda x: sales_predict(x, average_day_sale_dict, item_week_weekday), axis=1)

    return df[['id', 'sales']]


def main():
    train_df, test_df = sales_train_test()

    store_item_table = store_item_factor(train_df)

    annual_growth_dict = year_effect_by_item(train_df)
    item_week_weekday = weekday_effect(train_df)

    submission = sale_predict(test_df, annual_growth_dict, store_item_table, item_week_weekday)
    submission.to_csv('factor_model_weekday.csv', index=False)

if __name__ == "__main__":
    with timer("Full model run"):
        main()