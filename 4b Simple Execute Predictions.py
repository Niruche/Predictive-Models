#!/usr/bin/env python
# coding: utf-8

# In[11]:


# import packages
import pandas as pd
import numpy as np

from datetime import datetime, timedelta, time

import pyodbc
import urllib
import sqlalchemy as sa

from xgboost import XGBRegressor

import warnings
warnings.filterwarnings('ignore')

import pickle

quoted = urllib.parse.quote_plus("Driver={SQL Server Native Client 11.0};"
                      "Server=ssqlpaazu01;"
                      "Database=Pret_Predictive;"
                      "Trusted_Connection=yes;"
                      )
engine = sa.create_engine('mssql+pyodbc:///?odbc_connect={}'.format(quoted))

modelling_cols = ['shop_id','product_id','day_of_week_no','week_no_year','financial_month_no','weekend_flag',
                  'Same_day_1_week_ago','Same_day_2_week_ago','Same_day_3_week_ago','Same_day_4_week_ago',
                  'Same_day_5_week_ago','Same_day_6_week_ago','rolling_three_day_avg','rolling_three_day_max',
                  'rolling_three_day_min','rolling_five_day_avg','rolling_five_day_max','rolling_five_day_min',
                  'rolling_five_day_median','rolling_seven_day_median','actual_retail_sales_net','cluster']

if __name__ == "__main__":
    
    model_script = ''' select 
                            shop_id,
                            product_id,
                            t_date,
                            qty,
                            day_of_week_no,
                            week_no_year,
                            financial_month_no,
                            weekend_flag,
                            Same_day_1_week_ago,
                            Same_day_2_week_ago,
                            Same_day_3_week_ago,
                            Same_day_4_week_ago,
                            Same_day_5_week_ago,
                            Same_day_6_week_ago,
                            rolling_three_day_avg,
                            rolling_three_day_max,
                            rolling_three_day_min,
                            rolling_five_day_avg,
                            rolling_five_day_max,
                            rolling_five_day_min,
                            rolling_five_day_median,
                            rolling_seven_day_median,
                            actual_retail_sales_net,
                            cluster
                        from data.stage_PY_hot_bpd_feature_table_iteration_1
                        where t_date >= CAST(getdate() as date)  
                            '''

    model_df = pd.read_sql(model_script, engine)

    model_df['t_date'] = pd.to_datetime(model_df.t_date)

    model = pickle.load(open('F:\\Python_PROD\\LIVE Python SAV models\\Hot BPD Models\\Hot_BPD_model_iteration_1.sav', 'rb'))

    predictions = model.predict(model_df[modelling_cols])

    model_df['predictions'] = predictions

    model_df.to_csv('F:\\Python_PROD\\LIVE Python prediction CSV outputs\\Hot_bpd_predictions_iteration_1.csv', index = False)


# In[ ]:




