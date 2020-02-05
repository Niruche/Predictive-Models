
import pandas as pd
import numpy as np


pd.set_option('display.max_columns', None)

from datetime import datetime, timedelta, time

import pyodbc
import urllib
import sqlalchemy as sa

quoted = urllib.parse.quote_plus("Driver={SQL Server Native Client 11.0};"
                      "Server=ssqlpaazu01;"
                      "Database=Pret_Predictive;"
                      "Trusted_Connection=yes;"
                      )
engine = sa.create_engine('mssql+pyodbc:///?odbc_connect={}'.format(quoted))


# Imports
from xgboost import XGBRegressor

import pickle

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

# Define range finder function
def range_finder(df):

    ''' This function will determine whether the food item is currently being sold in the shops range of products'''

    # Make the groupby table to see number of days each items has been sold.  Note using count and not sum
    sales_quarter = df.groupby(['shop_id', 'product_id', 'quarter_label', 'weekend_flag'])[['qty']].count().rename(columns = {'qty' : 'no_days_sold'}).reset_index()

    # See how many sales days each shop has
    shop_open_days = df.groupby(['shop_id', 'quarter_label', 'weekend_flag'])[['t_date']].nunique().rename(columns = {'t_date' : 'shop_trading_days'}).reset_index()

    # Merge together
    sales_quarter = sales_quarter.merge(shop_open_days, on=['shop_id', 'quarter_label', 'weekend_flag'])

    # Now make a new column
    sales_quarter['sales_vol_percent'] = round(sales_quarter.no_days_sold / sales_quarter.shop_trading_days * 100, 1)

    # Make the column
    sales_quarter['out_of_range'] = sales_quarter.sales_vol_percent.map(lambda x: 1 if x <= 15 else 0)

    # now merge into main table
    df = df.merge(sales_quarter, on = ['shop_id', 'product_id', 'quarter_label', 'weekend_flag'])

    return df

# Write the function for outlier detetcion by row
def outlier_detector(row):

    # Initiate outlier value
    outlier = 0

    # Check the poss misskey flag
    if row['out_of_range'] == 0:

        # Now check conditions for outliers
        if (row['in_range_mean'] > 5) & (row['qty'] <= 3) :

            # Define value
            outlier = 1

        # Now check conditions for outliers
        elif (row['in_range_mean'] > 10) & (row['qty']/row['in_range_mean'] < 0.3) :

            # Define value
            outlier = 1

        # Next condition
        elif (row['in_range_mean'] <= 5) & (row['qty'] == 1):

            # Say this is an outlier
            outlier = 1

        # Next condition
        elif row['qty'] > 300:

            # Say this is an outlier
            outlier = 1

        # Otherwise say not an outlier
        else:
            outlier = 0

    # Define the other conditions
    else:

        # Rid of all 1 values
        if row['qty'] == 1:

            outlier = 1


        # Set up conditions - first if not sold in normal conditions ignore
        elif pd.isnull(row['in_range_mean']):

            outlier = 0

        # Now check conditions for outliers
        elif (row['in_range_mean'] > 10) & (row['qty']/row['in_range_mean'] < 0.3) :

            # Define value
            outlier = 1


        # Next condition
        elif row['qty'] / row['in_range_mean'] <= 0.15:

            outlier = 1

        else:
            outlier = 0

    return outlier

## Make a function to ad in the outlier column
def find_outliers(df):

    ''' Function that will find the outliers and add a column in the table'''

    # Make groupby table
    mean_range = df.groupby(['shop_id', 'product_id', 'week_day_description', 'out_of_range'])[['qty', 'no_days_sold']].mean().unstack()

    # Change column names
    mean_range.columns = ['in_range_mean', 'out_of_range_mean', 'days_sold_range', 'days_sold_out_of_range']

    # Merge into dataframe
    df = df.merge(mean_range, left_on=['shop_id', 'product_id', 'week_day_description'], right_index=True)

    # Now add the outlier columns
    df['outlier'] = df.apply(outlier_detector, axis=1)

    print('The outlier ratio for the table is {} percent'.format(df.loc[df.outlier == 1].shape[0]/ df.shape[0] * 100))

    return df

# feature builder
def mts_feature_builder(df, target = 'qty'):

    '''This function makes the features for the model

    Inputs:

    df:  This is the dataframe to make features from

    target: This is the target column we are building features from.  Default is 'qty' '''

    # First set the rolling average so is over the last week
    indexed_sales = df.set_index('t_date')

    # Groupby shop_id.  Only perform on actual sales column rolling back over 1 week.  REname the column too
    rolling_week = indexed_sales.groupby(['shop_id', 'product_id'])[[target]].rolling('7d', min_periods = 1).mean().rename(
        columns = {target : 'rolling_week_sales'})

    #rolling_std = indexed_sales.groupby(['shop_id', 'product_id'])[[target]].rolling('7d', min_periods = 1).std().rename(
        #columns = {target : 'rolling_std'})

    # Shift figures on one as rolling mean includes the figure from that day
    rolling_week = rolling_week.groupby(['shop_id', 'product_id']).shift(4).reset_index()
    #rolling_std = rolling_std.groupby(['shop_id', 'product_id']).shift(4).reset_index()

    # Merge into main data
    initial_rows = df.shape[0]
    df = df.merge(rolling_week[['shop_id', 'product_id', 't_date', 'rolling_week_sales']], on = ['shop_id', 'product_id', 't_date'])
    #df = df.merge(rolling_std[['shop_id', 'product_id', 't_date', 'rolling_std']], on = ['shop_id', 'product_id', 't_date'])
    final_rows = df.shape[0]

    if final_rows == initial_rows:
        print('OK to process')

    # create
    week_old = df.groupby(['shop_id', 'product_id'])[['rolling_week_sales']].shift(7).rename(columns = {'rolling_week_sales' : 'week_old_mean'})
    # Now add into table
    df['week_old'] = week_old

    # Make trend column
    df['weekly_trend'] = df.rolling_week_sales / df.week_old

    # Now look at the sales from last 3 similar weekdays
    rolling_4_day = df.groupby(['shop_id', 'product_id', 'day_of_week'])[[target]].rolling(4, min_periods = 1).mean().reset_index().set_index('level_3')
    rolling_6_day = df.groupby(['shop_id', 'product_id', 'day_of_week'])[[target]].rolling(6, min_periods = 1).mean().reset_index().set_index('level_3')


    # Can put this stright into df
    df['same_last_4_days'] = rolling_4_day.groupby(['shop_id', 'product_id', 'day_of_week']).shift()
    df['same_last_6_days'] = rolling_6_day.groupby(['shop_id', 'product_id', 'day_of_week']).shift()

    # Now look at the sales from last 3 similar weekdays
    rolling_4_week = df.groupby(['shop_id', 'product_id', 'weekend_flag'])[[target]].rolling(4, min_periods = 1).mean().reset_index().set_index('level_3')
    rolling_6_week = df.groupby(['shop_id', 'product_id', 'weekend_flag'])[[target]].rolling(6, min_periods = 1).mean().reset_index().set_index('level_3')

    # Now look at the max sales from last 4 and 6 days
    rolling_4_max = df.groupby(['shop_id', 'product_id', 'weekend_flag'])[[target]].rolling(4, min_periods = 1).max().reset_index().set_index('level_3')
    rolling_6_max = df.groupby(['shop_id', 'product_id', 'weekend_flag'])[[target]].rolling(6, min_periods = 1).max().reset_index().set_index('level_3')

    # Now look at the max sales from last 4 and 6 days
    rolling_4_min = df.groupby(['shop_id', 'product_id', 'weekend_flag'])[[target]].rolling(4, min_periods = 1).min().reset_index().set_index('level_3')
    rolling_6_min = df.groupby(['shop_id', 'product_id', 'weekend_flag'])[[target]].rolling(6, min_periods = 1).min().reset_index().set_index('level_3')

    # Split out by weekday and weekend
    rolling_4_weekday = rolling_4_week.loc[rolling_4_week.weekend_flag == 0].groupby(['shop_id', 'product_id'])[[target]].shift(4)
    rolling_4_weekend = rolling_4_week.loc[rolling_4_week.weekend_flag == 1].groupby(['shop_id', 'product_id'])[[target]].shift(2)

    rolling_6_weekday = rolling_6_week.loc[rolling_6_week.weekend_flag == 0].groupby(['shop_id', 'product_id'])[[target]].shift(4)
    rolling_6_weekend = rolling_6_week.loc[rolling_6_week.weekend_flag == 1].groupby(['shop_id', 'product_id'])[[target]].shift(2)

    # Max
    rolling_4_max_weekday = rolling_4_max.loc[rolling_4_max.weekend_flag == 0].groupby(['shop_id', 'product_id'])[[target]].shift(4)
    rolling_4_max_weekend = rolling_4_max.loc[rolling_4_max.weekend_flag == 1].groupby(['shop_id', 'product_id'])[[target]].shift(2)

    rolling_6_max_weekday = rolling_6_max.loc[rolling_6_max.weekend_flag == 0].groupby(['shop_id', 'product_id'])[[target]].shift(4)
    rolling_6_max_weekend = rolling_6_max.loc[rolling_6_max.weekend_flag == 1].groupby(['shop_id', 'product_id'])[[target]].shift(2)

    # Min
    rolling_4_min_weekday = rolling_4_min.loc[rolling_4_min.weekend_flag == 0].groupby(['shop_id', 'product_id'])[[target]].shift(4)
    rolling_4_min_weekend = rolling_4_min.loc[rolling_4_min.weekend_flag == 1].groupby(['shop_id', 'product_id'])[[target]].shift(2)

    rolling_6_min_weekday = rolling_6_min.loc[rolling_6_min.weekend_flag == 0].groupby(['shop_id', 'product_id'])[[target]].shift(4)
    rolling_6_min_weekend = rolling_6_min.loc[rolling_6_min.weekend_flag == 1].groupby(['shop_id', 'product_id'])[[target]].shift(2)



    # Can put this stright into df
    df['rolling_4_days'] = pd.concat([rolling_4_weekday, rolling_4_weekend])
    df['rolling_6_days'] = pd.concat([rolling_6_weekday, rolling_6_weekend])

    # Max
    df['rolling_4_max'] = pd.concat([rolling_4_max_weekday, rolling_4_max_weekend])
    df['rolling_6_max'] = pd.concat([rolling_6_max_weekday, rolling_6_max_weekend])

    # Min
    df['rolling_4_min'] = pd.concat([rolling_4_min_weekday, rolling_4_min_weekend])
    df['rolling_6_min'] = pd.concat([rolling_6_min_weekday, rolling_6_min_weekend])


    # Fill in all columns with actual sales for now
    df.rolling_week_sales = df.rolling_week_sales.fillna(df[target])
    df.same_last_4_days = df.same_last_4_days.fillna(df[target])
    df.same_last_6_days = df.same_last_6_days.fillna(df[target])
    df.rolling_4_days = df.rolling_4_days.fillna(df[target])
    df.rolling_6_days = df.rolling_6_days.fillna(df[target])
    # Fill columns
    df.rolling_4_max = df.rolling_4_max.fillna(df.rolling_4_days)
    df.rolling_4_min = df.rolling_4_min.fillna(df.rolling_4_days)

    df.rolling_6_max = df.rolling_6_max.fillna(df.rolling_6_days)
    df.rolling_6_min = df.rolling_6_min.fillna(df.rolling_6_days)


    # Weekly trend
    df.weekly_trend = df.weekly_trend.fillna(1)

    # daily trend
    df['daily_trend'] = df.same_last_4_days / df.same_last_6_days

    df.daily_trend = df.daily_trend.fillna(1)

    # Rid ourselves of infinite numbers
    df.weekly_trend = df.weekly_trend.map(lambda x: 1 if x == np.inf else x)
    df.daily_trend = df.daily_trend.map(lambda x: 1 if x == np.inf else x)

    # Take a look
    return df

# Now create a function to map in the weather
def weather_map(df):

    '''This function puts the weather features onto the dataframe'''

    # Load mapping table
    weather_map = pd.read_sql('select Shop_ID, SiteUID from data.stage_weather_shop_site_mapping', engine)

    # Merge in teh mapping
    df = df.merge(weather_map, left_on='shop_id', right_on='Shop_ID', how = 'left')

    # Load the table of forecast weather
    forecast_weather = pd.read_sql('select * from data.stage_weather_forecast_agg', engine)

    # Filter down
    forecast_weather = forecast_weather.loc[forecast_weather.CurrentForecast == 1].copy()

    # Now change to datetime
    forecast_weather.ForecastDate = pd.to_datetime(forecast_weather.ForecastDate)

    # Make a list of useful columns,
    useful_weather_cols = ['ForecastDate', 'MaxTemperature', 'MinTemperature', 'FeelsLikeTemperature', 'SunshineHours',
                           'WeatherDescriptor', 'ForecastSiteUID']

    #Merge it in
    df = df.merge(forecast_weather[useful_weather_cols], left_on=['SiteUID', 't_date'], right_on=['ForecastSiteUID', 'ForecastDate'], how = 'left')

    # Take a look at the weather values - need cleaning as have a line break in them
    df.WeatherDescriptor = df.WeatherDescriptor.map(lambda x: x.replace('\r', '') if pd.notnull(x) else x)

    # Make a weather label encoding
    descriptor_dict = {}

    for i, weather_cond in enumerate(df.WeatherDescriptor.value_counts().index):

        descriptor_dict[weather_cond] = i

    # Make the column
    df['Weather_Code'] = df.WeatherDescriptor.map(lambda x: descriptor_dict[x] if pd.notnull(x) else x)

    # Put the temperatures into bins
    bins = [-15, -10, -5, 0, 5, 10, 15, 20, 25, 30]
    names = ['-15 to -10', '-10 to -5', '-5 to 0', '0 to 5', '5 to 10', '10 to 15', '15 to 20', '20 to 25', '25 to 30']

    df['Temp Range'] = pd.cut(df['FeelsLikeTemperature'], bins, labels=names)

    # Make a dict
    dict_weather_map = dict(zip(names, bins[:-1]))

    # Now use to map column
    df['temp_code'] = df['Temp Range'].map(lambda x: dict_weather_map[x])

    # Convert to float
    df['temp_code'] = df['temp_code'].astype(float)

    # Drop the SHop_ID column for saving to SQL
    df = df.drop('Shop_ID', axis = 1)

    return df

# Develop a class to test the models
class Model_Tester():

    '''Class object to test and train a model'''

    def __init__(self, feature_columns, target_col = 'qty', cols_to_remove = ['qty', 't_date',
                                                                                             'in_range_mean', 'outlier',
                                                                                            'product_name', 'category_desc'],
                train_end_date = '20190101'):

        self.feature_columns = feature_columns
        self.target_col = target_col
        self.cols_to_remove = cols_to_remove
        self.train_end_date = train_end_date

    def table_filler(self, df, fill_zero):

        if fill_zero == True:

            df['SameDay_1_weeks'] = df['SameDay_1_weeks'].fillna(0)
            df['SameDay_2_weeks'] = df['SameDay_2_weeks'].fillna(0)
            df['SameDay_3_weeks'] = df['SameDay_3_weeks'].fillna(0)
            df['SameDay_4_weeks'] = df['SameDay_4_weeks'].fillna(0)
            df['SameDay_5_weeks'] = df['SameDay_5_weeks'].fillna(0)
            df['SameDay_6_weeks'] = df['SameDay_6_weeks'].fillna(0)

        else:

            df['SameDay_1_weeks'] = df['SameDay_1_weeks'].fillna(df['qty'])
            df['SameDay_2_weeks'] = df['SameDay_2_weeks'].fillna(df['SameDay_1_weeks'])
            df['SameDay_3_weeks'] = df['SameDay_3_weeks'].fillna(df['SameDay_2_weeks'])
            df['SameDay_4_weeks'] = df['SameDay_4_weeks'].fillna(df['SameDay_3_weeks'])
            df['SameDay_5_weeks'] = df['SameDay_5_weeks'].fillna(df['SameDay_4_weeks'])
            df['SameDay_6_weeks'] = df['SameDay_6_weeks'].fillna(df['SameDay_5_weeks'])


        return df

    # Make a method to score accuracy
    def accuracy_scorer(self, row, target_column = 'qty'):

        ''' A function for measuring accuracy

        This function is applied over a dataframe.  Can also define the target column to measure against.

        Currently set as 'qty' '''

        # Define output
        accuracy = 0

        # First look at qty
        if row[self.target_col] <= 20:

            if abs(row[target_column] - row['predictions']) <= 2:

                accuracy = 1

            else:
                accuracy = 0

        # Now look at large values
        else:

            if abs(row[target_column] - row['predictions']) / row[target_column] <= 0.1:

                accuracy = 1

            else:
                accuracy = 0

        return accuracy



    def train_and_test(self, df, fill_zero = False, clean_table = False, round_val = True):

        # First prepare table
        if clean_table == True:
            clean_df = self.table_filler(df, fill_zero)
        else:
            clean_df = df.copy()

        # Make modelling table
        clean_df = clean_df.loc[clean_df.t_date >= '20180101', self.feature_columns]

        # Now set up training and test sets
        train_df = clean_df.loc[clean_df.t_date < self.train_end_date]
        test_df = clean_df.loc[(clean_df.t_date >= self.train_end_date) & (clean_df.t_date < '20190610')]

        # Set modelling columns
        self.modelling_cols = [col for col in self.feature_columns if col not in self.cols_to_remove]

        # Set Train and test sets
        X_train = train_df[self.modelling_cols]
        y_train = train_df[self.target_col]

        X_test = test_df[self.modelling_cols]
        y_test = test_df[self.target_col]

        # Set up model
        self.xg_model = XGBRegressor(random_state=42)

        # Fit data
        self.xg_model.fit(X_train, y_train)

        # Predictions
        self.pred = self.xg_model.predict(X_test)

        # Score
        self.score = self.xg_model.score(X_test, y_test)

        # Now put in predictions and get MAPE
        test_df['predictions'] = self.pred

        # Now for rounding
        if round_val == True:
            test_df['predictions'] = test_df['predictions'].map(lambda x: round(x, 0))

        # Now MAPe column
        test_df['MAPE'] = abs(test_df[self.target_col] - test_df.predictions) / test_df[self.target_col]

        # Accuracy column
        test_df['Accuracy'] = test_df.apply(lambda x: self.accuracy_scorer(x, self.target_col), axis = 1)

        # Get MAPE score
        self.MAPE_score = test_df.loc[test_df.qty >= 3, 'MAPE'].mean()

        # Accuracy Score
        self.Accuracy_score = test_df.loc[test_df.qty >= 3, 'Accuracy'].mean()

        return test_df


def new_accuracy_score_new_qty(row, target_pred = 'weekend_predictions'):

    '''This function goes over a dataframe.  Can define the prediction column to measure against'''

    # Define output
    accuracy = 0

    # First look at qty
    if row['qty'] <= 20:

        if abs(row['qty'] - row[target_pred]) <= 2:

            accuracy = 1

        else:
            accuracy = 0

    # Now look at large values
    else:

        if abs(row['qty'] - row[target_pred]) / row['qty'] <= 0.1:

            accuracy = 1

        else:
            accuracy = 0

    return accuracy

# Develop a class to test the models
class Model_Tester_plus():

    '''Class object to test and train a model'''

    def __init__(self, feature_columns, target_col = 'qty', cols_to_remove = ['qty', 't_date',
                                                                                             'in_range_mean', 'outlier',
                                                                                            'product_name', 'category_desc'],
                train_end_date = '20190101', test_end_date = '20190630'):

        self.feature_columns = feature_columns
        self.target_col = target_col
        self.cols_to_remove = cols_to_remove
        self.train_end_date = train_end_date
        self.test_end_date = test_end_date

    def table_filler(self, df, fill_zero):

        if fill_zero == True:

            df['SameDay_1_weeks'] = df['SameDay_1_weeks'].fillna(0)
            df['SameDay_2_weeks'] = df['SameDay_2_weeks'].fillna(0)
            df['SameDay_3_weeks'] = df['SameDay_3_weeks'].fillna(0)
            df['SameDay_4_weeks'] = df['SameDay_4_weeks'].fillna(0)
            df['SameDay_5_weeks'] = df['SameDay_5_weeks'].fillna(0)
            df['SameDay_6_weeks'] = df['SameDay_6_weeks'].fillna(0)

        else:

            df['SameDay_1_weeks'] = df['SameDay_1_weeks'].fillna(df['qty'])
            df['SameDay_2_weeks'] = df['SameDay_2_weeks'].fillna(df['SameDay_1_weeks'])
            df['SameDay_3_weeks'] = df['SameDay_3_weeks'].fillna(df['SameDay_2_weeks'])
            df['SameDay_4_weeks'] = df['SameDay_4_weeks'].fillna(df['SameDay_3_weeks'])
            df['SameDay_5_weeks'] = df['SameDay_5_weeks'].fillna(df['SameDay_4_weeks'])
            df['SameDay_6_weeks'] = df['SameDay_6_weeks'].fillna(df['SameDay_5_weeks'])


        return df

    # Make a method to score accuracy
    def accuracy_scorer(self, row, target_column = 'qty'):

        ''' A function for measuring accuracy

        This function is applied over a dataframe.  Can also define the target column to measure against.

        Currently set as 'qty' '''

        # Define output
        accuracy = 0

        # First look at qty
        if row[self.target_col] <= 20:

            if abs(row[target_column] - row['predictions']) <= 2:

                accuracy = 1

            else:
                accuracy = 0

        # Now look at large values
        else:

            if abs(row[target_column] - row['predictions']) / row[target_column] <= 0.1:

                accuracy = 1

            else:
                accuracy = 0

        return accuracy



    def train_and_test(self, df, fill_zero = False, clean_table = False, round_val = True):

        # First prepare table
        if clean_table == True:
            clean_df = self.table_filler(df, fill_zero)
        else:
            clean_df = df.copy()

        # Make modelling table
        clean_df = clean_df.loc[clean_df.t_date >= '20180101', self.feature_columns]

        # Now set up training and test sets
        train_df = clean_df.loc[clean_df.t_date < self.train_end_date]
        test_df = clean_df.loc[(clean_df.t_date >= self.train_end_date) & (clean_df.t_date <= self.test_end_date)]

        # Set modelling columns
        self.modelling_cols = [col for col in self.feature_columns if col not in self.cols_to_remove]

        # Set Train and test sets
        X_train = train_df[self.modelling_cols]
        y_train = train_df[self.target_col]

        X_test = test_df[self.modelling_cols]
        y_test = test_df[self.target_col]

        # Set up model
        self.xg_model = XGBRegressor(random_state=42)

        # Fit data
        self.xg_model.fit(X_train, y_train)

        # Predictions
        self.pred = self.xg_model.predict(X_test)

        # Score
        self.score = self.xg_model.score(X_test, y_test)

        # Now put in predictions and get MAPE
        test_df['predictions'] = self.pred

        # Now for rounding
        if round_val == True:
            test_df['predictions'] = test_df['predictions'].map(lambda x: round(x, 0))

        # Now MAPe column
        test_df['MAPE'] = abs(test_df[self.target_col] - test_df.predictions) / test_df[self.target_col]

        # Accuracy column
        test_df['Accuracy'] = test_df.apply(lambda x: self.accuracy_scorer(x, self.target_col), axis = 1)

        # Get MAPE score
        self.MAPE_score = test_df.loc[test_df.qty >= 3, 'MAPE'].mean()

        # Accuracy Score
        self.Accuracy_score = test_df.loc[test_df.qty >= 3, 'Accuracy'].mean()

        return test_df
