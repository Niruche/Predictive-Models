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

# First define functions
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

def anomaly_build(): 

    # Load in the Table
    food_q_script = '''

                SELECT
                    a.*,
                    B.actual_retail_sales_net,
                    d.cluster,
                    d.description,
                    E.LAUNCH_NAME AS quarter_label
                FROM DATA.stage_till_data_MTS_agg as a
                    LEFT JOIN DATA.stage_sales_history_day_aggregate AS B ON A.shop_id = B.shop_id AND a.t_date = B.date
                    LEFT JOIN [DATA].[PY_SALES_FORECAST_CLUSTERS] as d on d.shop_id = A.shop_id
                    LEFT JOIN (
                                SELECT DISTINCT
                                    A.DATE,
                                    B.LAUNCH_NAME
                                FROM data.stage_calendar AS A
                                    INNER JOIN (
                                                select
                                                    launch_name,
                                                    START_DATE,
                                                    END_DATE
                                                from warehouse.range_tool.dbo.[vw_launch_date] as a
                                            ) AS B ON CAST(A.date AS DATE) BETWEEN CAST(B.START_DATE AS DATE) AND CAST(B.END_DATE AS date)
                                ) AS E ON E.date = a.t_date
                WHERE
                    t_date BETWEEN dateadd(mm,-6,cast(getdate() as date)) AND dateadd(dd,-1,cast(getdate() as date))
    '''

    mts_initial = pd.read_sql(food_q_script, engine)
    mts_initial = range_finder(mts_initial)
    mts_initial = find_outliers(mts_initial)
    
    mts_initial = mts_initial[['shop_id','t_date','product_id','qty','outlier' ]]
    
    return mts_initial.to_csv('F:\\Python_PROD\\LIVE Python generated feature table CSV\\MTS_anomaly_detector.csv' , index = False )

if __name__ == '__main__':
    
    anomaly_build()

