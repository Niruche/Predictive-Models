#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import packages
import pandas as pd
import numpy as np

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

from xgboost import XGBRegressor

import warnings
warnings.filterwarnings('ignore')

import pickle


def cluster_model_predictor():
    
    model_script = '''    
                    select
                        a.shop_id,
                        product_id, 
                        a.t_date,
                        weekend_flag, 
                        rolling_week_sales,
                        weekly_trend,
                        same_last_4_days, 
                        same_last_6_days,
                        daily_trend, 
                        0 as out_of_range, 
                        rolling_4_days, 
                        rolling_6_days, 
                        day_of_week_no, 
                        rolling_4_max, 
                        rolling_6_max,
                        rolling_4_min, 
                        rolling_6_min, 
                        bank_holiday_flag, 
                        isnull(b.school_holiday,0) as school_holiday, 
                        Weather_Code, 
                        temp_code,
                        c.cluster
                    from [DATA].[stage_PY_MTS_feature_table_iteration_2] as a
                        left join ( select distinct shop_id, date, school_holiday from stage_PY_sales_forecast_features ) 
                        as b on a.shop_id = b.shop_id and a.t_date = b.date
                        LEFT JOIN [DATA].[PY_SALES_FORECAST_CLUSTERS] as c on c.shop_id = A.shop_id
                    where t_date >= CAST(getdate() as date)   
                    '''
    
    model_df = pd.read_sql(model_script, engine)
    
    # Test using CSV file
    # model_df = pd.read_csv('.\\test_tables\\all_shops_sales_model_feature_table.csv', sep = '\t')
    
    # Now change to datetime
    model_df.date = pd.to_datetime(model_df.t_date)
    
    # Make a model dictionary
    model_dict = {}

    for i in range(6):

        model_dict[i] = pickle.load(open('F:\\Python_PROD\\LIVE Python SAV models\\MTS Models\\Cluster_' + str(i) + '_Allday_MTS_Model.sav', 'rb'))
        
    # Initialise an output table
    output_table = pd.DataFrame()
    
    # DEfine modelling cols
    modelling_cols = ['shop_id','product_id', 'weekend_flag', 'rolling_week_sales', 'weekly_trend', 'same_last_4_days', 'same_last_6_days',
                      'daily_trend', 'out_of_range', 'rolling_4_days', 'rolling_6_days', 'day_of_week_no', 'rolling_4_max','rolling_6_max',
                      'rolling_4_min', 'rolling_6_min', 'bank_holiday_flag', 'school_holiday', 'Weather_Code', 'temp_code']

    for cluster in range(6):

        prediction_table = model_df.loc[model_df.cluster == cluster, modelling_cols]

        # Predictions
        pred = model_dict[cluster].predict(prediction_table)

        # Make a temporary table
        temp_cluster = model_df.loc[prediction_table.index]

        # Now put in predictions
        temp_cluster['prediction'] = pred

        output_table = output_table.append(temp_cluster)

    output_table = output_table.sort_index()
    
    output_table = output_table[['shop_id',	'product_id',	't_date',	'weekend_flag',	'rolling_week_sales',	'weekly_trend',
                                 'same_last_4_days',	'same_last_6_days',	'daily_trend',	'out_of_range',	'rolling_4_days',
                                 'rolling_6_days',	'day_of_week_no',	'rolling_4_max',	'rolling_6_max',	'rolling_4_min',
                                 'rolling_6_min',	'bank_holiday_flag',	'school_holiday',	'Weather_Code',	'temp_code','cluster',
                                 'prediction'
                                ]]
    
    output_table.to_csv('F:\\Python_PROD\\LIVE Python prediction CSV outputs\\MTS All day predictions iteration 3.csv', index = False, sep = '\t')
    
if __name__ == '__main__':

    cluster_model_predictor()

