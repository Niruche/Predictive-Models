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

def mts_feature_builder(df, target = 'qty'):

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

if __name__ == "__main__":

    base_table = pd.read_sql('select * from vw_MTS_base_table_build_iteration_2_stage', engine)

    # Covert to datetime
    base_table.t_date = pd.to_datetime(base_table.t_date)

    # Add in new qty columns
    base_table['new_qty'] = base_table.qty.map(lambda x: 4 if x <= 3 else x)

    # Get rid out find_outliers
    base_table = base_table.loc[base_table.outlier == 0]

    # Now make features
    feature_table = mts_feature_builder(base_table, target = 'new_qty')

    # Now add in weather
    feature_table = weather_map(feature_table)
    
    feature_table = feature_table[['shop_id','t_date','weekend_flag','day_of_week_no','day_of_week','bank_holiday_flag',
                                   'week_day_description','product_id','qty','actual_retail_sales_net','cluster','description',
                                   'quarter_label','outlier','new_qty','rolling_week_sales','week_old','weekly_trend','same_last_4_days',
                                   'same_last_6_days','rolling_4_days','rolling_6_days','rolling_4_max','rolling_6_max','rolling_4_min',
                                   'rolling_6_min','daily_trend','SiteUID','ForecastDate','MaxTemperature','MinTemperature',
                                   'FeelsLikeTemperature','SunshineHours','WeatherDescriptor','ForecastSiteUID','Weather_Code',
                                   'Temp Range','temp_code'
                                  ]]
    
    feature_table.to_csv('F:\\Python_PROD\\LIVE Python generated feature table CSV\\MTS_basic_feature_table_iteration_2.csv', index = False)
