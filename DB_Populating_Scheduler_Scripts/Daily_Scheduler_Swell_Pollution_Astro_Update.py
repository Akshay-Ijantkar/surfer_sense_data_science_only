#!/usr/bin/env python
# coding: utf-8

# ### Author: Akshay Ijantkar
# ### Team: Aqua Wizards
# ### Project: Surfers Bible

# * https://launchschool.com/books/sql/read/table_relationships

# # Import Libraries:

# 0 1 * * * /usr/bin/python3 /home/ubuntu/pop_db_sch_ss/Daily_Scheduler_Swell_Pollution_Astro_News_API.py >> /home/ubuntu/pop_db_sch_ss/log_Daily_Scheduler_Swell_Pollution_Astro_News_API.txt 2>&1

# In[2]:


import pandas as pd
import numpy as np
import random
import matplotlib as plt
from matplotlib import pyplot
# import seaborn as sns; sns.set()
# from scipy.stats import norm 
import matplotlib.pyplot as plt
# For Linear regression
# from sklearn.linear_model import LinearRegression
# For split given dataset into train and test set.
# from sklearn.model_selection import train_test_split
# To verify models using this metrics 
# from sklearn.metrics import mean_squared_error, r2_score

# import statsmodels.formula.api as smf
# from sklearn.linear_model import LinearRegression
# from sklearn import metrics
# v
from matplotlib import rcParams
# rcParams['figure.figsize'] = 50,50
# import pandas_profiling
pd.set_option('display.max_rows', 1500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from pandas import ExcelWriter
from pandas import ExcelFile

# from pygeocoder import Geocoder

import sys
# from weather_au import api
# from weather_au import summary
# from weather import place, observations, uv_index
import time

import json
import requests
from datetime import datetime, timedelta
# from catboost import CatBoostClassifier
# from sklearn.model_selection import train_test_split

# from sklearn.metrics import r2_score
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasRegressor
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# from sklearn.model_selection import cross_validate

# import catboost as ctb
# from catboost import CatBoostRegressor, FeaturesData, Pool
# from scipy.stats import uniform as sp_randFloat
# from scipy.stats import randint as sp_randInt
# from scipy.stats import uniform
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.metrics.pairwise import cosine_similarity
# from  sklearn.metrics.pairwise import euclidean_distances
# from sklearn.metrics.pairwise import manhattan_distances
# from sklearn.metrics.pairwise import pairwise_distances
import re
# import pprint
from datetime import date
import datetime
# import sqlite3
# from sqlite3 import Error
from datetime import datetime
from dateutil import tz
import datetime

# get_ipython().run_line_magic('matplotlib', 'inline')


# # Give Date:

# In[63]:


from datetime import datetime
import datetime
no_days_from_today = 3

select_date = ""

if select_date == "":    
    today = date.today()
    today_date = today.strftime("%Y-%m-%d") 
    given_date =  str((datetime.datetime.strptime(today_date, "%Y-%m-%d") + datetime.timedelta(days = no_days_from_today)).date())
    print("today_date = ", today_date)
    print("given_date = ", given_date)
else:
    given_date = select_date
    print("given_date = ", given_date)
# print("today_date = ", today_date)


# # Import Beach Table to DF: 

# In[6]:


import psycopg2 as ps

# define credentials 
credentials = {'POSTGRES_ADDRESS' : 'test-surfers-bible-instance.cljoljhkgpfb.ap-southeast-2.rds.amazonaws.com', # change to your endpoint
               'POSTGRES_PORT' : 5432, # change to your port
               'POSTGRES_USERNAME' : '', # change to your username
               'POSTGRES_PASSWORD' : '', # change to your password
               'POSTGRES_DBNAME' : 'test_surfers_bible_db'} # change to your db name

# create connection and cursor    
conn = ps.connect(host=credentials['POSTGRES_ADDRESS'],
                  database=credentials['POSTGRES_DBNAME'],
                  user=credentials['POSTGRES_USERNAME'],
                  password=credentials['POSTGRES_PASSWORD'],
                  port=credentials['POSTGRES_PORT'])

cur = conn.cursor()

beach_df = pd.read_sql_query("SELECT * FROM BEACH_TABLE;", conn)

conn.close()

cur.close()

# beach_df.head()


# # API KEYS:

# In[7]:


#  In Production API KEYS:
SG_API_KEY_DICT = {}
SG_API_KEY_DICT['1'] = "" #JIA
SG_API_KEY_DICT['2'] = "" #JIA
SG_API_KEY_DICT['3'] = "" #JIA
SG_API_KEY_DICT['4'] = "" #JIA

SG_API_KEY_DICT['5'] = "" #JIA
SG_API_KEY_DICT['6'] = "" #JIA
SG_API_KEY_DICT['7'] = "" # MY
SG_API_KEY_DICT['8'] = "" # MY

SG_API_KEY_DICT['9'] = ""  #Nishant
SG_API_KEY_DICT['10'] = "" #Nishant
SG_API_KEY_DICT['11'] = "" #Nishant
SG_API_KEY_DICT['12'] = "" #Nishant

# SG_API_KEY_DICT['test'] = "" # My

# API KEYS in TESTING
# SG_API_KEY_DICT = {}
# SG_API_KEY_DICT[" # JIA
# SG_API_KEY_DICT['2'] = "" # MY
# SG_API_KEY_DICT['3'] = "" # MY
# SG_API_KEY_DICT['4'] = "" # My 

# SG_API_KEY_DICT['5'] = "" #SAT
# SG_API_KEY_DICT['6'] = "" #SAT
# SG_API_KEY_DICT['7'] = "" # JIA 
# SG_API_KEY_DICT['8'] = "" # JIA

# SG_API_KEY_DICT['9'] =  "2" # JIA
# SG_API_KEY_DICT['10'] = "" # JIA
# SG_API_KEY_DICT['11'] = "" # JIA
# SG_API_KEY_DICT['12'] = "" # JIA

# SG_API_KEY_DICT['test_2'] = "" 


# # UTILITY FUNCTIONS:

# In[8]:


def get_unix_time_epochs(datetime_str):
    return int(time.mktime(time.strptime(str(datetime_str), '%Y-%m-%dT%H:%M:%S')))


# In[9]:


def convert_datetime_in_dif_timezones(from_datetime_str, from_timezone_str, to_timezone_str, 
                                      datetime_format = '%Y-%m-%dT%H:%M:%S'):
#     USE pytz.all_timezones to get all timestamps
    from datetime import datetime
    import pytz
    date_time_obj = datetime.strptime(from_datetime_str, datetime_format)
#     print("date_time_obj = ", date_time_obj)
    
    old_timezone = pytz.timezone(from_timezone_str)
    new_timezone = pytz.timezone(to_timezone_str)
    
    new_timezone_timestamp = old_timezone.localize(date_time_obj).astimezone(new_timezone).strftime("%Y-%m-%dT%H:%M:%S") 
#     print("new_timezone_timestamp", new_timezone_timestamp)
    return str(new_timezone_timestamp)
    


# # SWELL DATA:

# # SWELL: API CALLS:

# In[28]:


swell_col_lst = [
    'date',
    'beach_id',
    'beach_name',
    'beach_latitude',
    'beach_longitude',
    'beach_state',    
    "beach_municipality",
    
    'airTemperature',
    'cloudCover',
    'currentDirection',
    'currentSpeed',
    'gust',
    'humidity',
    'precipitation',
    'pressure',
    'seaLevel',
    'secondarySwellDirection',
    'secondarySwellHeight',
    'secondarySwellPeriod',
    'swellDirection',
    'swellHeight',
    'swellPeriod',
    'time',
    'visibility',
    'waterTemperature',
    'waveDirection',
    'waveHeight',
    'wavePeriod',
    'windDirection',
    'windSpeed',
    'windWaveDirection',
    'windWaveHeight',
    'windWavePeriod'
]
swell_df = pd.DataFrame(columns = swell_col_lst
                         )
# swell_df


# In[30]:


# %%time
# Wall time: 5min 48s
import datetime
for api_limit, api_no in zip(range(0,200,50) ,range(5,9)):
    print("++++++++++++++++++++++++++++++++++++++++++++")
    API_KEY = SG_API_KEY_DICT[str(api_no)]
    print("SWELL: api_limit = ", api_limit)
    print("api_no = ", api_no)
    print("++++++++++++++++++++++++++++++++++++++++++++")
    
    for index, row in beach_df.loc[api_limit  : api_limit + 50,].iterrows():
        print("")
        print("index = ", index)
        print("beach_name", row["beach_name"])
        print(" ")

        latitude = row['beach_latitude']
        longitude = row['beach_longitude']

        start_date = given_date
        end_date = str((datetime.datetime.strptime(given_date, "%Y-%m-%d") + datetime.timedelta(days = 1)).date())
#         end_date = '2020-06-07'
        
        url_str = "https://api.stormglass.io/v2/weather/point?"
        url_str += "lat="+ str(latitude) +"&"
        url_str += "lng="+str(longitude) + "&"
        url_str += "start=" + start_date + "&"
        url_str += "end="+ end_date + "&"
        url_str += "params=waterTemperature,wavePeriod,waveDirection,waveHeight,windWaveDirection,windWaveHeight,windWavePeriod,swellPeriod,secondarySwellPeriod,swellDirection,secondarySwellDirection,swellHeight,secondarySwellHeight,windSpeed,windDirection,airTemperature,precipitation,gust,cloudCover,humidity,pressure,visibility,currentSpeed,currentDirection,seaLevel"

        payload = {}
        headers = {
          'Authorization': API_KEY
        }

        response = requests.request("GET", url_str, headers=headers, data = payload)
        response_dict = json.loads(response.text)    
    #     print("response_dict = ", response_dict)
        swell_dict = {}

        for hour_dict in response_dict['hours']:
    #         print("hour_dict = ", hour_dict)

            swell_dict['date'] = given_date
            for beach_col in beach_df.columns.tolist():
                swell_dict[beach_col] = row[beach_col]     

            swell_col_lst = list(hour_dict.keys())
            swell_col_lst.remove("time")

            for swell_col in swell_col_lst:

                swell_dict[swell_col] = hour_dict[swell_col]['sg']

            swell_dict['time'] = hour_dict['time']

            swell_df = swell_df.append(swell_dict, ignore_index=True)
###################################################################################################################


# # CONVERT UTC TIME TO AUSTRALIA TIME:

# In[31]:


swell_df['time'] = swell_df['time'].apply(lambda x: convert_datetime_in_dif_timezones(from_datetime_str = x, 
                                                                                  from_timezone_str = 'UTC', 
                                                                                  to_timezone_str ='Australia/Melbourne', 
                                                                                datetime_format = '%Y-%m-%dT%H:%M:%S+00:00'))


# In[41]:


def missing_imputation_rf_func(df, 
                               feature_lst = [
                                                'beach_latitude', 
                                                'beach_longitude', 
                                                'airTemperature', 
                                                'cloudCover', 
                                                'gust', 
                                                'humidity', 
                                                'precipitation', 
                                                'pressure',
                                                'waterTemperature', 
                                                'waveDirection', 
                                                'waveHeight', 
                                                'wavePeriod',
                                           ],
                              target_lst = [
                                                'waterTemperature', 
                                                'waveDirection', 
                                                'waveHeight', 
                                                'wavePeriod',                               
                                          ]):
    miss_impu_df = df[feature_lst].copy()    
    from missingpy import MissForest
    imputer = MissForest()
    X_imputed = imputer.fit_transform(miss_impu_df.values)
    aft_miss_imp_df = pd.DataFrame(
                      X_imputed, 
                      columns = feature_lst) 
    for col in target_lst:    
        df[col] = aft_miss_imp_df[col].values
        df[col] = df[col].apply(lambda x: round(x, 2))
    return df


# In[44]:


try:
    swell_df = missing_imputation_rf_func(swell_df)
except:
    pass


# In[45]:


try:
    swell_df['watertemperature'].fillna((swell_df['watertemperature'].mean()), inplace=True)
    swell_df['wavedirection'].fillna((swell_df['wavedirection'].mean()), inplace=True)
    swell_df['waveHeight'].fillna((swell_df['waveHeight'].mean()), inplace=True)
    swell_df['wavePeriod'].fillna((swell_df['wavePeriod'].mean()), inplace=True)
except:
    pass


# # CREATE SEA_WATER_QUALITY_TABLE:

# In[46]:


import psycopg2 as ps

# define credentials 
credentials = {'POSTGRES_ADDRESS' : 'test-surfers-bible-instance.cljoljhkgpfb.ap-southeast-2.rds.amazonaws.com', # change to your endpoint
               'POSTGRES_PORT' : 5432, # change to your port
               'POSTGRES_USERNAME' : 'ai_postgres', # change to your username
               'POSTGRES_PASSWORD' : '', # change to your password
               'POSTGRES_DBNAME' : 'test_surfers_bible_db'} # change to your db name

# create connection and cursor    
conn = ps.connect(host=credentials['POSTGRES_ADDRESS'],
                  database=credentials['POSTGRES_DBNAME'],
                  user=credentials['POSTGRES_USERNAME'],
                  password=credentials['POSTGRES_PASSWORD'],
                  port=credentials['POSTGRES_PORT'])

cur = conn.cursor()

# cur.execute("DROP TABLE SEA_WATER_QUALITY_TABLE;")

create_table_query = '''
      CREATE TABLE IF NOT EXISTS SWELL_TABLE
      (
        swell_id SERIAL PRIMARY KEY,
        date DATE,
        beach_id INTEGER,
        beach_name TEXT,
        beach_latitude REAL,
        beach_longitude REAL,
        beach_state TEXT,
        beach_municipality TEXT,
        airTemperature REAL,
        cloudCover REAL,
        currentDirection REAL,
        currentSpeed REAL,
        gust REAL,
        humidity REAL,
        precipitation REAL,
        pressure REAL,
        seaLevel REAL,
        secondarySwellDirection REAL,
        secondarySwellHeight REAL,
        secondarySwellPeriod REAL,
        swellDirection REAL,
        swellHeight REAL,
        swellPeriod REAL,
        time TEXT,
        visibility REAL,
        waterTemperature REAL,
        waveDirection REAL,
        waveHeight REAL,
        wavePeriod REAL,
        windDirection REAL,
        windSpeed REAL,
        windWaveDirection REAL,
        windWaveHeight REAL,
        windWavePeriod REAL,
        FOREIGN KEY (beach_id) REFERENCES beach_table(beach_id) ON DELETE CASCADE
       ); 
       '''

cur.execute(create_table_query)
conn.commit()


# # INSERT SEA_WATER_QUALITY_TABLE:

# In[47]:


# %%time
# Wall time: 12min 51s
import psycopg2 as ps

# define credentials 
credentials = {'POSTGRES_ADDRESS' : 'test-surfers-bible-instance.cljoljhkgpfb.ap-southeast-2.rds.amazonaws.com', # change to your endpoint
               'POSTGRES_PORT' : 5432, # change to your port
               'POSTGRES_USERNAME' : 'ai_postgres', # change to your username
               'POSTGRES_PASSWORD' : '', # change to your password
               'POSTGRES_DBNAME' : 'test_surfers_bible_db'} # change to your db name

# create connection and cursor    
conn = ps.connect(host=credentials['POSTGRES_ADDRESS'],
                  database=credentials['POSTGRES_DBNAME'],
                  user=credentials['POSTGRES_USERNAME'],
                  password=credentials['POSTGRES_PASSWORD'],
                  port=credentials['POSTGRES_PORT'])

cur = conn.cursor()

fill_question_mark_str = str(tuple(["%s"  for i in swell_df.columns.tolist()])).replace("'", "")
fill_question_mark_str

for row in swell_df.itertuples():
    data_tuple = tuple(row[1:])

    print("data_tuple = ", data_tuple)
    print(" ")
    
    cur.execute("""
                        INSERT INTO SWELL_TABLE
                        (
                            date,
                            beach_id,
                            beach_name,
                            beach_latitude,
                            beach_longitude,
                            beach_state,
                            beach_municipality,
                            airTemperature,
                            cloudCover,
                            currentDirection,
                            currentSpeed,
                            gust,
                            humidity,
                            precipitation,
                            pressure,
                            seaLevel,
                            secondarySwellDirection,
                            secondarySwellHeight,
                            secondarySwellPeriod,
                            swellDirection,
                            swellHeight,
                            swellPeriod,
                            time,
                            visibility,
                            waterTemperature,
                            waveDirection,
                            waveHeight,
                            wavePeriod,
                            windDirection,
                            windSpeed,
                            windWaveDirection,
                            windWaveHeight,
                            windWavePeriod
                         ) VALUES  
                         """ + fill_question_mark_str + " ;"
                , data_tuple)    

conn.commit()
cur.close()
conn.close()


# # ASTRONOMICAL DATA:

# # astronomy: API CALLS:

# In[54]:


astronomy_col_lst = [
    'date',
    'beach_id',
    'beach_name',
    'beach_latitude',
    'beach_longitude',
    'beach_state',    
    "beach_municipality",
    
    'astronomicalDawn',
     'astronomicalDusk',
     'civilDawn',
     'civilDusk',
     'moonFraction',
     'moonPhase',
     'moonrise',
     'moonset',
     'nauticalDawn',
     'nauticalDusk',
     'sunrise',
     'sunset',
     'time'
]
astronomy_df = pd.DataFrame(columns = astronomy_col_lst
                         )
astronomy_df


# In[56]:


# %%time
# Wall time: 4min 34s
import datetime
for api_limit, api_no in zip(range(0,200,50) ,range(9,13)):
    
    print("++++++++++++++++++++++++++++++++++++++++++++")
    API_KEY = SG_API_KEY_DICT[str(api_no)]
    print("ASTRONOMY: api_limit = ", api_limit)
    print("api_no = ", api_no)
    print("++++++++++++++++++++++++++++++++++++++++++++")
    
    for index, row in beach_df.loc[api_limit  : api_limit + 50,].iterrows():

        print("")
        print("index = ", index)
        print("beach_name", row["beach_name"])
        print(" ")

        latitude = row['beach_latitude']
        longitude = row['beach_longitude']

        start_date = given_date
        end_date = str((datetime.datetime.strptime(given_date, "%Y-%m-%d") + datetime.timedelta(days = 1)).date())
#         end_date = '2020-06-07'

        url_str = "https://api.stormglass.io/v2/astronomy/point?"
        url_str += "lat="+ str(latitude) +"&"
        url_str += "lng="+str(longitude) + "&"
        url_str += "start=" + start_date + "&"
        url_str += "end="+ end_date 
    #         + "&"
    #         url_str += "params=waterTemperature,wavePeriod,waveDirection,waveHeight,windWaveDirection,windWaveHeight,windWavePeriod,astronomyPeriod,secondaryastronomyPeriod,astronomyDirection,secondaryastronomyDirection,astronomyHeight,secondaryastronomyHeight,windSpeed,windDirection,airTemperature,precipitation,gust,cloudCover,humidity,pressure,visibility,currentSpeed,currentDirection,seaLevel"

        payload = {}
        headers = {
          'Authorization': API_KEY
        }
        try:
            response = requests.request("GET", url_str, headers=headers, data = payload)
            response_dict = json.loads(response.text)    
        #     print("response_dict = ", response_dict)
            astronomy_dict = {}

            for hour_dict in response_dict['data']:
        #         print("hour_dict = ", hour_dict)

                astronomy_dict['date'] = given_date

                for beach_col in beach_df.columns.tolist():
                    astronomy_dict[beach_col] = row[beach_col]     

                astronomy_col_lst = list(hour_dict.keys())
        #             astronomy_col_lst.remove("time")

                for astronomy_col in astronomy_col_lst:

                    astronomy_dict[astronomy_col] = hour_dict[astronomy_col]

        #             astronomy_dict['time'] = hour_dict['time']

                astronomy_df = astronomy_df.append(astronomy_dict, ignore_index=True)
        except:
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("FAILED REQUEST!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("api_limit = ", api_limit)
            print("api_no = ", api_no)
            print("index = ", index)
            print("beach_name", row["beach_name"])            
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
###################################################################################################################


# # CONVERT UTC TIME TO AUSTRALIA TIME:

# In[57]:


def convert_datetime_in_dif_timezones(from_datetime_str, from_timezone_str, to_timezone_str, 
                                      datetime_format = '%Y-%m-%dT%H:%M:%S'):
#     USE pytz.all_timezones to get all timestamps
    from datetime import datetime
    import pytz
    try:
        date_time_obj = datetime.strptime(from_datetime_str, datetime_format)
    #     print("date_time_obj = ", date_time_obj)

        old_timezone = pytz.timezone(from_timezone_str)
        new_timezone = pytz.timezone(to_timezone_str)

        new_timezone_timestamp = old_timezone.localize(date_time_obj).astimezone(new_timezone).strftime("%Y-%m-%dT%H:%M:%S") 
    #     print("new_timezone_timestamp", new_timezone_timestamp)
        return str(new_timezone_timestamp)
    except:
        print("from_datetime_str",  from_datetime_str)


# In[58]:


time_col_lst = ['astronomicalDawn', 'astronomicalDusk', 'civilDawn', 'civilDusk', 'moonrise', 'moonset',
                'nauticalDawn', 'nauticalDusk', 'sunrise', 'sunset', 'time']

for time_col in time_col_lst: 
    astronomy_df[time_col] = astronomy_df[time_col].apply(lambda x: convert_datetime_in_dif_timezones(
                                                                                from_datetime_str = x, 
                                                                                from_timezone_str = 'UTC', 
                                                                                to_timezone_str ='Australia/Melbourne', 
                                                                                datetime_format = '%Y-%m-%dT%H:%M:%S+00:00'))


# In[59]:


astronomy_df.drop("moonPhase", axis=1, inplace = True)


# # CREATE astronomy_TABLE:

# In[60]:


import psycopg2 as ps

# define credentials 
credentials = {'POSTGRES_ADDRESS' : 'test-surfers-bible-instance.cljoljhkgpfb.ap-southeast-2.rds.amazonaws.com', # change to your endpoint
               'POSTGRES_PORT' : 5432, # change to your port
               'POSTGRES_USERNAME' : 'ai_postgres', # change to your username
               'POSTGRES_PASSWORD' : '', # change to your password
               'POSTGRES_DBNAME' : 'test_surfers_bible_db'} # change to your db name

# create connection and cursor    
conn = ps.connect(host=credentials['POSTGRES_ADDRESS'],
                  database=credentials['POSTGRES_DBNAME'],
                  user=credentials['POSTGRES_USERNAME'],
                  password=credentials['POSTGRES_PASSWORD'],
                  port=credentials['POSTGRES_PORT'])

cur = conn.cursor()

# cur.execute("DROP TABLE SEA_WATER_QUALITY_TABLE;")

create_table_query = '''
      CREATE TABLE IF NOT EXISTS astronomy_TABLE
      (
        astronomy_id SERIAL PRIMARY KEY,
        date DATE,
        beach_id INTEGER,
        beach_name TEXT,
        beach_latitude REAL,
        beach_longitude REAL,
        beach_state TEXT,
        beach_municipality TEXT,
        astronomicalDawn TEXT,
        astronomicalDusk TEXT,
        civilDawn TEXT,
        civilDusk TEXT,
        moonFraction REAL,
        moonrise TEXT,
        moonset TEXT,
        nauticalDawn TEXT,
        nauticalDusk TEXT,
        sunrise TEXT,
        sunset TEXT,
        time TEXT,
        FOREIGN KEY (beach_id) REFERENCES beach_table(beach_id) ON DELETE CASCADE
       ); 
       '''

cur.execute(create_table_query)
conn.commit()


# # INSERT astronomy_TABLE:

# In[61]:


# %%time
# Wall time: 12min 51s
import psycopg2 as ps

# define credentials 
credentials = {'POSTGRES_ADDRESS' : 'test-surfers-bible-instance.cljoljhkgpfb.ap-southeast-2.rds.amazonaws.com', # change to your endpoint
               'POSTGRES_PORT' : 5432, # change to your port
               'POSTGRES_USERNAME' : 'ai_postgres', # change to your username
               'POSTGRES_PASSWORD' : '', # change to your password
               'POSTGRES_DBNAME' : 'test_surfers_bible_db'} # change to your db name

# create connection and cursor    
conn = ps.connect(host=credentials['POSTGRES_ADDRESS'],
                  database=credentials['POSTGRES_DBNAME'],
                  user=credentials['POSTGRES_USERNAME'],
                  password=credentials['POSTGRES_PASSWORD'],
                  port=credentials['POSTGRES_PORT'])

cur = conn.cursor()

fill_question_mark_str = str(tuple(["%s"  for i in astronomy_df.columns.tolist()])).replace("'", "")
fill_question_mark_str

for row in astronomy_df.itertuples():
    data_tuple = tuple(row[1:])

    print("data_tuple = ", data_tuple)
    print(" ")
    
    cur.execute("""
                        INSERT INTO astronomy_TABLE
                        (
                            date,
                            beach_id,
                            beach_name,
                            beach_latitude,
                            beach_longitude,
                            beach_state,
                            beach_municipality,
                            astronomicalDawn,
                            astronomicalDusk,
                            civilDawn,
                            civilDusk,
                            moonFraction,
                            moonrise,
                            moonset,
                            nauticalDawn,
                            nauticalDusk,
                            sunrise,
                            sunset,
                            time
                         ) VALUES  
                         """ + fill_question_mark_str + " ;"
                , data_tuple)    

conn.commit()
cur.close()
conn.close()


# ___
# ___
# ___
# 

# # SEA_WATER_QUALITY DATA:

# In[10]:


bio_col_lst = [  
                'date',
                'beach_id',
                'beach_name',
                'beach_latitude',
                'beach_longitude',
                'beach_state',    
                "beach_municipality",
    
                 'chlorophyll',
                 'iron',
                 'nitrate',
                 'oxygen',
                 'ph',
                 'phosphate',
                 'phyto',
                 'phytoplankton',
                 'salinity',
                 'silicate',
                 'time'
                ]

bio_df = pd.DataFrame(columns = bio_col_lst
                         )
# bio_df


# # SEA_WATER_QUALITY: API CALLS:

# In[13]:


# %%time
# Wall time: 5min 34s
import datetime
for api_limit, api_no in zip(range(0,200,50) ,range(1,5)):
    print("++++++++++++++++++++++++++++++++++++++++++++")
    API_KEY = SG_API_KEY_DICT[str(api_no)]
    print("api_limit = ", api_limit)
    print("api_no = ", api_no)
    print("++++++++++++++++++++++++++++++++++++++++++++")
    
    for index, row in beach_df.loc[api_limit  : api_limit + 50,].iterrows():
        print("")
        print("index = ", index)
        print("beach_name", row["beach_name"])
        print(" ")

        latitude = row['beach_latitude']
        longitude = row['beach_longitude']

        start_date = given_date
#         end_date = str((datetime.datetime.strptime(given_date, "%Y-%m-%d") + datetime.timedelta(days = 1)).date())
        end_date = '2020-06-06'


        url_str = "https://api.stormglass.io/v2/bio/point?"
        url_str += "lat="+ str(latitude) +"&"
        url_str += "lng="+str(longitude) + "&"
        url_str += "start=" + start_date + "&"
        url_str += "end="+ end_date + "&"
        url_str += "params=iron,nitrate,chlorophyll,phyto,oxygen,ph,phytoplankton,phosphate,silicate,salinity"

        payload = {}
        headers = {
          'Authorization': API_KEY
        }

        response = requests.request("GET", url_str, headers=headers, data = payload)
        response_dict = json.loads(response.text)    
        print("response_dict = ", response_dict)
        bio_dict = {}

        for hour_dict in response_dict['hours']:
    #         print("hour_dict = ", hour_dict)

            bio_dict['date'] = given_date
            for beach_col in beach_df.columns.tolist():
                bio_dict[beach_col] = row[beach_col]     

            bio_col_lst = list(hour_dict.keys())
            bio_col_lst.remove("time")

            for bio_col in bio_col_lst:

                bio_dict[bio_col] = hour_dict[bio_col]['sg']

            bio_dict['time'] = hour_dict['time']

            bio_df = bio_df.append(bio_dict, ignore_index=True)
###################################################################################################################


# # CONVERT UTC TIME TO AUSTRALIA TIME:

# In[18]:


bio_df['time'] = bio_df['time'].apply(lambda x: convert_datetime_in_dif_timezones(from_datetime_str = x, 
                                                                                  from_timezone_str = 'UTC', 
                                                                                  to_timezone_str ='Australia/Melbourne', 
                                                                                datetime_format = '%Y-%m-%dT%H:%M:%S+00:00'))


# In[25]:


# bio_df.isnull().sum()
try:
    bio_df['ph'].fillna((bio_df['ph'].mean()), inplace=True)
except:
    pass


# # CREATE SEA_WATER_QUALITY_TABLE:

# In[26]:


import psycopg2 as ps

# define credentials 
credentials = {'POSTGRES_ADDRESS' : 'test-surfers-bible-instance.cljoljhkgpfb.ap-southeast-2.rds.amazonaws.com', # change to your endpoint
               'POSTGRES_PORT' : 5432, # change to your port
               'POSTGRES_USERNAME' : 'ai_postgres', # change to your username
               'POSTGRES_PASSWORD' : '', # change to your password
               'POSTGRES_DBNAME' : 'test_surfers_bible_db'} # change to your db name

# create connection and cursor    
conn = ps.connect(host=credentials['POSTGRES_ADDRESS'],
                  database=credentials['POSTGRES_DBNAME'],
                  user=credentials['POSTGRES_USERNAME'],
                  password=credentials['POSTGRES_PASSWORD'],
                  port=credentials['POSTGRES_PORT'])

cur = conn.cursor()

# cur.execute("DROP TABLE SEA_WATER_QUALITY_TABLE;")

create_table_query = '''
      CREATE TABLE IF NOT EXISTS SEA_WATER_QUALITY_TABLE
      (
        sea_water_quality_id SERIAL PRIMARY KEY,
        date DATE,
        beach_id INTEGER,
        beach_name TEXT,
        beach_latitude REAL,
        beach_longitude REAL,
        beach_state  TEXT,
        beach_municipality TEXT,
        chlorophyll REAL,
        iron REAL,
        nitrate REAL,
        oxygen REAL,
        ph REAL,
        phosphate REAL,
        phyto REAL,
        phytoplankton REAL,
        salinity REAL,
        silicate REAL,
        time TEXT,
        FOREIGN KEY (beach_id) REFERENCES beach_table(beach_id) ON DELETE CASCADE
       ); 
       '''

cur.execute(create_table_query)
conn.commit()


# # INSERT SEA_WATER_QUALITY_TABLE:

# In[27]:


# %%time
# Wall time: 12min 51s
import psycopg2 as ps

# define credentials 
credentials = {'POSTGRES_ADDRESS' : 'test-surfers-bible-instance.cljoljhkgpfb.ap-southeast-2.rds.amazonaws.com', # change to your endpoint
               'POSTGRES_PORT' : 5432, # change to your port
               'POSTGRES_USERNAME' : 'ai_postgres', # change to your username
               'POSTGRES_PASSWORD' : '', # change to your password
               'POSTGRES_DBNAME' : 'test_surfers_bible_db'} # change to your db name

# create connection and cursor    
conn = ps.connect(host=credentials['POSTGRES_ADDRESS'],
                  database=credentials['POSTGRES_DBNAME'],
                  user=credentials['POSTGRES_USERNAME'],
                  password=credentials['POSTGRES_PASSWORD'],
                  port=credentials['POSTGRES_PORT'])

cur = conn.cursor()

fill_question_mark_str = str(tuple(["%s"  for i in bio_df.columns.tolist()])).replace("'", "")
fill_question_mark_str

for row in bio_df.itertuples():
    data_tuple = tuple(row[1:])

    print("data_tuple = ", data_tuple)
    print(" ")
    
    cur.execute("""
                        INSERT INTO SEA_WATER_QUALITY_TABLE
                        (
                            date,
                            beach_id,
                            beach_name,
                            beach_latitude,
                            beach_longitude,
                            beach_state,
                            beach_municipality,
                            chlorophyll,
                            iron,
                            nitrate,
                            oxygen,
                            ph,
                            phosphate,
                            phyto,
                            phytoplankton,
                            salinity,
                            silicate,
                            time
                         ) VALUES  
                         """ + fill_question_mark_str + " ;"
                , data_tuple)    

conn.commit()
cur.close()
conn.close()


# ___
# ___
# ___

# In[ ]:




