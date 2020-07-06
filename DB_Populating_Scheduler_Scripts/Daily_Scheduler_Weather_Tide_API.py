#!/usr/bin/env python
# coding: utf-8

# ### Author: Akshay Ijantkar
# ### Team: Aqua Wizards
# ### Project: Surfers Bible

# * https://launchschool.com/books/sql/read/table_relationships

# # CRONTAB EXPRESSIONA and STATEMENT:

# 0 2 * * * /usr/bin/python3 /home/ubuntu/pop_db_sch_ss/Daily_Scheduler_Weather_Tide_API.py >> /home/ubuntu/pop_db_sch_ss/log_Daily_Scheduler_Weather_Tide_API.txt 2>&1

# In[ ]:


# log_Daily_Scheduler_Weather_Tide_API


# # Import Libraries:

# In[1]:


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
from sklearn import metrics
# v
from matplotlib import rcParams
# rcParams['figure.figsize'] = 50,50
# import pandas_profiling
# pd.set_option('display.max_rows', 1500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
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
from catboost import CatBoostClassifier
# from sklearn.model_selection import train_test_split

# from sklearn.metrics import r2_score
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasRegressor
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# from sklearn.model_selection import cross_validate

import catboost as ctb
# from catboost import CatBoostRegressor, FeaturesData, Pool
# from scipy.stats import uniform as sp_randFloat
# from scipy.stats import randint as sp_randInt
# from scipy.stats import uniform
# from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics.pairwise import cosine_similarity
# from  sklearn.metrics.pairwise import euclidean_distances
# from sklearn.metrics.pairwise import manhattan_distances
# from sklearn.metrics.pairwise import pairwise_distances
import re
import pprint
from datetime import date
import datetime
# import sqlite3
# from sqlite3 import Error
from datetime import datetime
from dateutil import tz

# %matplotlib inline


# # Decide Date:

# In[2]:


from datetime import datetime
import datetime

no_days_from_today = 3

select_date = ""

if select_date == "":    
    today = date.today()
    today_date = today.strftime("%Y-%m-%d") 
    given_date =  str((datetime.datetime.strptime(today_date, "%Y-%m-%d") + datetime.timedelta(days = no_days_from_today)).date())
    given_date_timestamp_epoch = int(time.mktime(time.strptime(given_date, '%Y-%m-%d')))
else:
    given_date = select_date
    given_date_timestamp_epoch = int(time.mktime(time.strptime(given_date, '%Y-%m-%d')))

print("today_date = ", today_date)
print("given_date = ", given_date)
print("given_date_timestamp_epoch = ",given_date_timestamp_epoch)


# In[3]:


# date_5days_lst = []
# for day in range(5):
#     date_5days_lst.append(str((datetime.datetime.strptime(given_date, "%Y-%m-%d") + datetime.timedelta(days = day)).date()))
# date_5days_lst    


# In[4]:


# timestamp_epochs_5days_lst = []
# for date in date_5days_lst:
# #     + 3600
#     timestamp_epochs_5days_lst.append(int(time.mktime(time.strptime(date, '%Y-%m-%d'))) )
# timestamp_epochs_5days_lst    


# # READ BEACH DATABASE FROM RDS POSTGRES DB:

# In[5]:


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


# # Import Beach Table to DF: 

# In[6]:


beach_df = pd.read_sql_query("SELECT * FROM BEACH_TABLE;", conn)
# beach_df.head()


# In[7]:


# beach_df.columns.tolist()


# # Weather API:

# In[8]:


weather_beach_req_col_lst = ['date',
 'beach_id',
 'beach_name',
 'beach_latitude',
 'beach_longitude',
 'beach_state',
 'time',
 'summary',
 'icon',
 'precipIntensity',
 'precipProbability',
 'temperature',
 'apparentTemperature',
 'dewPoint',
 'humidity',
 'pressure',
 'windSpeed',
 'windGust',
 'windBearing',
 'cloudCover',
 'uvIndex',
 'visibility',
 'ozone']

weather_df = pd.DataFrame(columns = weather_beach_req_col_lst
                         )
# weather_df  


# In[9]:


# %%time
# Wall time: 24min 24s
API_KEY = ""
cnt = 0
# https://api.darksky.net/forecast//-33.869,151.209,2019-12-30T12:00:00

# for date in date_5days_lst[:]:
    
#     timestamp_epochs = int(time.mktime(time.strptime(date, '%Y-%m-%d')))

for index, row in beach_df.loc[:,].iterrows():

    print("beach_name = ", row["beach_name"])
    print("date = ",given_date,"\n")
    print("index = ", index)

    get_request = ""
    get_request += "https://api.darksky.net/forecast/"
    get_request += API_KEY
    get_request += "/"
    get_request += str(row["beach_latitude"])
    get_request += ","
    get_request += str(row["beach_longitude"])
    get_request += ","
    get_request += str(given_date_timestamp_epoch)

    response_dict = json.loads(requests.get(get_request).text)

    print("get_request = ", get_request)
    print("****************************************************************************************************")
#     print("response_dict = ",response_dict)

    if len(list(response_dict.keys())) > 2:
        cnt = cnt + 1
        print("cnt =>>>>>>>>>>>>>>>>>>>>> ", cnt)

        weather_row_dict = {}

        for per_hr_attri_dict in response_dict['hourly']['data']:

            weather_row_dict["date"] = given_date

            for col in ['beach_id', 'beach_name', 'beach_latitude', 'beach_longitude', 'beach_state']:
                weather_row_dict[col] = row[col]

            for hr_attri in per_hr_attri_dict.keys():
                weather_row_dict[hr_attri] = per_hr_attri_dict[hr_attri]                    

            weather_row_dict["nearest-station"] = response_dict['flags']['nearest-station']
            weather_row_dict["sources"] = response_dict['flags']['sources']
            weather_row_dict["offset"] = response_dict['offset']            

            weather_df = weather_df.append(weather_row_dict, ignore_index=True)


    else:
        print("############################################################################################")
        print("Response Failed...!")
        print("beach_name = ", row["beach_name"])
        print("date = ",date,"\n")
        print("index = ", index)
        print("############################################################################################")
#     time.sleep(1)
# weather_df            


# # Convert timestamp epochs to datetime format:

# In[10]:


weather_df["datetime"] = weather_df["time"].apply(lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x)))


# * https://launchschool.com/books/sql/read/table_relationships

# In[12]:


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

create_table_query = '''
      CREATE TABLE IF NOT EXISTS WEATHER_TABLE
      (
        weather_id SERIAL PRIMARY KEY,
        date DATE,
        beach_id INTEGER NOT NULL,
        beach_name TEXT,
        beach_latitude REAL,
        beach_longitude REAL,
        beach_state TEXT,
        time REAL,
        summary TEXT,
        icon TEXT,
        precipIntensity REAL,
        precipProbability REAL,
        temperature REAL,
        apparentTemperature REAL,
        dewPoint REAL,
        humidity REAL,
        pressure REAL,
        windSpeed REAL,
        windGust REAL,
        windBearing REAL,
        cloudCover REAL,
        uvIndex REAL,
        visibility REAL,
        ozone REAL,
        nearest_station REAL,
        time_offset REAL,
        precipType TEXT,
        sources TEXT,
        datetime TEXT,
        FOREIGN KEY (beach_id) REFERENCES beach_table(beach_id) ON DELETE CASCADE
       ); 
       '''

cur.execute(create_table_query)
conn.commit()

cur.close()
conn.close()


# In[13]:


# %%time
# Wall time: 12min 16s
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

fill_question_mark_str = str(tuple(["%s"  for i in weather_df.columns.tolist()])).replace("'", "")
fill_question_mark_str

for row in weather_df.itertuples():
    data_tuple = tuple(row[1:])

#     print("data_tuple = ", data_tuple)
#     print(" ")
    
    cur.execute("""
                        INSERT INTO WEATHER_TABLE
                        (
                            date,
                            beach_id,
                            beach_name,
                            beach_latitude,
                            beach_longitude,
                            beach_state,
                            time,
                            summary,
                            icon,
                            precipIntensity,
                            precipProbability,
                            temperature,
                            apparentTemperature,
                            dewPoint,
                            humidity,
                            pressure,
                            windSpeed,
                            windGust,
                            windBearing,
                            cloudCover,
                            uvIndex,
                            visibility,
                            ozone,
                            nearest_station,
                            time_offset,
                            precipType,
                            sources,
                            datetime
                         ) VALUES  
                         """ + fill_question_mark_str + " ;"
                , data_tuple)    

conn.commit()

cur.close()
conn.close()


# In[14]:


weather_df["date_beach_name"] = weather_df[
                                                ['date', 'beach_name']
                                             ].apply(lambda x: '|'.join(x.astype(str).values), axis=1)
# weather_df.head()


# In[15]:


# log_dataset_path = "D:\Monash_University_Stuff\Final_Semester\IE\Surfers_Bible_Code_Commit\Datasets\Daily_weather_tide_log\\"
# weather_df.to_csv(log_dataset_path + "weather_df"+"_10_05"+".csv")


# ## AGGREGATE WEATHER:

# In[16]:


agg_ops_lst = ['max','min','mean','median','std','var','sem']
weather_numeric_features = ["temperature", "apparentTemperature", "dewPoint", "humidity", "windSpeed", "windBearing", "uvIndex",
                    "cloudCover"]

agg_dict = dict(zip(weather_numeric_features, [agg_ops_lst for i in range(len(weather_numeric_features))]))
# agg_dict


# In[17]:


for col in weather_numeric_features:
    weather_df[col] = weather_df[col].astype(float) 


# In[18]:


weather_beach_req_col_lst = [
    "date_beach_name",
    'date',
     'beach_name',
     'beach_latitude',
     'beach_longitude',
     'beach_state',
]

agg_weather_df = weather_df[weather_numeric_features + ["date_beach_name"]].groupby('date_beach_name').agg(agg_dict)

agg_weather_df.columns = ["_".join(x) for x in agg_weather_df.columns.ravel()]
agg_weather_df.reset_index(level=0, inplace=True)
agg_weather_df =  pd.merge(
                           agg_weather_df, 
                           weather_df[weather_beach_req_col_lst],
                           left_on = "date_beach_name",
                           right_on = "date_beach_name",
                           how = "inner"
                            )

agg_weather_df.drop_duplicates(inplace = True)
agg_weather_df.reset_index(inplace = True, 
                           drop = True)

print("agg_weather_df.shape = ", agg_weather_df.shape)
# agg_weather_df.head()


# In[19]:


# agg_weather_df.isnull().sum()


# In[20]:


# log_dataset_path = "D:\Monash_University_Stuff\Final_Semester\IE\Surfers_Bible_Code_Commit\Datasets\Daily_weather_tide_log\\"
# agg_weather_df.to_csv(log_dataset_path + "agg_weather_df"+"_10_05"+".csv")


# # END OF WEATHER API PROCESSING:
# 
# ---
# ---
# ---

# # TIDE API PROCESSING STARTS:

# In[23]:


heights_tide_df = pd.DataFrame(columns = [
                                         'date',
                                        'beach_id',
                                        'beach_name',
                                        'beach_latitude',
                                        'beach_longitude',
                                        'beach_state',
                                         'timestamp', 
                                         'datetime', 
                                         'height', 
                                         'state', 
                                         'origin_distance',
                                         "origin_distance_unit", 
                                         "origin_latitude", 
                                         "origin_longitude", 
                                         "timezone"
                                        ]
                              )

# heights_tide_df 
extremes_tide_df = pd.DataFrame(columns = [
                                        'date',
                                        'beach_id',
                                        'beach_name',
                                        'beach_latitude',
                                        'beach_longitude',
                                        'beach_state',   
                                        'timestamp', 
                                        'datetime', 
                                        'height', 
                                        'state'
                                          ]
                               )

# extremes_tide_df
datum_tide_df = pd.DataFrame(columns = [
                                        'date',
                                        'beach_id',
                                        'beach_name',
                                        'beach_latitude',
                                        'beach_longitude',
                                        'beach_state',                
                                        'timestamp', 
                                        'datetime', 
                                        'datum', 
                                        'LAT', 
                                        'HAT'
                                        ]
                            )


# In[24]:


# %%time
# Wall time: 14min 4s
# datum_tide_df
cnt = 0
for index, row in beach_df.loc[:,:].iterrows():
#     if index >= 0:
    print("beach_name = ", row["beach_name"])
    print("index = ", index)
    timestamp_epochs = int(time.mktime(time.strptime(given_date, '%Y-%m-%d')))

    url = "https://tides.p.rapidapi.com/tides"    
    querystring = {"interval":"60",
                   "duration":"1440",
#                    "duration": "7200",
                   "timestamp": str(timestamp_epochs),
                   "latitude":str(row["beach_latitude"]),
                   "longitude":str(row["beach_longitude"])}

    headers = {
        'x-rapidapi-host': "tides.p.rapidapi.com",
        'x-rapidapi-key': "",
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'
        }

    response_dict = json.loads(requests.request("GET", url, headers = headers, params = querystring).text)
#     print("response_dict - ", response_dict)
    print(" ")
    if response_dict['status'] == 200:
        cnt = cnt + 1
        print("cnt = ", cnt)

#         Updating heights_tide_df 
#                                          'date',
#                                         'beach_id',
#                                         'beach_name',
#                                         'beach_latitude',
#                                         'beach_longitude',
#                                         'beach_state',  
#                                          'timestamp', 
#                                          'datetime', 
#                                          'height', 
#                                          'state', 
#                                          'origin_distance',
#                                          "origin_distance_unit", 
#                                          "origin_latitude", 
#                                          "origin_longitude", 
#                                          "timezone"

        heights_tide_row_dict = {}
        heights_tide_row_dict["date"] = given_date
        heights_tide_row_dict["beach_id"] = row["beach_id"]
        heights_tide_row_dict["beach_name"] = row["beach_name"]
        heights_tide_row_dict["beach_latitude"] = row["beach_latitude"]
        heights_tide_row_dict["beach_longitude"] = row["beach_longitude"]
        heights_tide_row_dict["beach_state"] = row["beach_state"]
        

        heights_tide_row_dict["origin_distance"] = response_dict['origin']['distance']
        heights_tide_row_dict["origin_distance_unit"] = response_dict['origin']['unit']
        heights_tide_row_dict["origin_latitude"] = response_dict['origin']['latitude']
        heights_tide_row_dict["origin_longitude"] = response_dict['origin']['longitude']
        heights_tide_row_dict["timezone"] = response_dict["timezone"]

        for ht_dict in response_dict['heights']:
            heights_tide_row_dict["timestamp"] = ht_dict['timestamp']
            heights_tide_row_dict["datetime"] = ht_dict['datetime']
            heights_tide_row_dict["height"] = ht_dict['height']
            heights_tide_row_dict["state"] = ht_dict['state']

            heights_tide_df = heights_tide_df.append(heights_tide_row_dict, 
                                                 ignore_index = True)

#         Updating extremes_tide_df:
#                                         'date',
#                                         'beach_address',
#                                         'beach_name',
#                                         'country_state',
#                                         'country',
#                                         'latitude',
#                                         'longitude',    
#                                         'timestamp', 
#                                         'datetime', 
#                                         'height', 
#                                         'state'

        extremes_tide_row_dict = {}
        extremes_tide_row_dict["date"] = given_date
        extremes_tide_row_dict["beach_id"] = row["beach_id"]
        extremes_tide_row_dict["beach_name"] = row["beach_name"]
        extremes_tide_row_dict["beach_latitude"] = row["beach_latitude"]
        extremes_tide_row_dict["beach_longitude"] = row["beach_longitude"]
        extremes_tide_row_dict["beach_state"] = row["beach_state"] 

        for extremes_dict in response_dict['extremes']:

            extremes_tide_row_dict["timestamp"] = extremes_dict['timestamp']
            extremes_tide_row_dict["datetime"] = extremes_dict['datetime']
            extremes_tide_row_dict["height"] = extremes_dict['height']
            extremes_tide_row_dict["state"] = extremes_dict['state']   

            extremes_tide_df = extremes_tide_df.append(extremes_tide_row_dict, 
                                                 ignore_index = True)

        #         Updating datum_tide_df
#                                         'date',
#                                         'beach_address',
#                                         'beach_name',
#                                         'country_state',
#                                         'country',
#                                         'latitude',
#                                         'longitude',                
#                                         'timestamp', 
#                                         'datetime', 
#                                         'datum', 
#                                         'LAT', 
#                                         'HAT'
        datum_tide_row_dict = {}
        datum_tide_row_dict["date"] = given_date
        datum_tide_row_dict["beach_id"] = row["beach_id"]
        datum_tide_row_dict["beach_name"] = row["beach_name"]
        datum_tide_row_dict["beach_latitude"] = row["beach_latitude"]
        datum_tide_row_dict["beach_longitude"] = row["beach_longitude"]
        datum_tide_row_dict["beach_state"] = row["beach_state"] 

        datum_tide_row_dict["timestamp"] = response_dict["timestamp"]
        datum_tide_row_dict["datetime"] = response_dict["datetime"]
        datum_tide_row_dict["datum"] = response_dict["datum"]
        datum_tide_row_dict["LAT"] = response_dict['datums']["LAT"]
        datum_tide_row_dict["HAT"] = response_dict['datums']["HAT"]

        datum_tide_df = datum_tide_df.append(
                                            datum_tide_row_dict,
                                            ignore_index = True
                                            )
        print(" ")
        time.sleep(1)


# In[30]:


print("heights_tide_df.shape = ",heights_tide_df.shape)
# heights_tide_df.head()


# In[31]:


# log_dataset_path = "D:\Monash_University_Stuff\Final_Semester\IE\Surfers_Bible_Code_Commit\Datasets\Daily_weather_tide_log\\"
# heights_tide_df.to_csv(log_dataset_path + "heights_tide_df"+"_10_05"+".csv")


# In[32]:


print("extremes_tide_df.shape = ", extremes_tide_df.shape)
# extremes_tide_df.head()


# In[33]:


# log_dataset_path = "D:\Monash_University_Stuff\Final_Semester\IE\Surfers_Bible_Code_Commit\Datasets\Daily_weather_tide_log\\"
# extremes_tide_df.to_csv(log_dataset_path + "extremes_tide_df"+"_10_05"+".csv")


# ## CONVERT UTC TIME to LOCAL TIME:

# In[34]:


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
    


# In[35]:


heights_tide_df["datetime"] = heights_tide_df["datetime"].apply(lambda x: convert_datetime_in_dif_timezones(from_datetime_str = x, 
                                                                                  from_timezone_str = 'UTC', 
                                                                                  to_timezone_str ='Australia/Melbourne', 
                                                                                datetime_format = '%Y-%m-%dT%H:%M:%S+00:00'))

extremes_tide_df["datetime"] = extremes_tide_df["datetime"].apply(lambda x: convert_datetime_in_dif_timezones(from_datetime_str = x, 
                                                                                  from_timezone_str = 'UTC', 
                                                                                  to_timezone_str ='Australia/Melbourne', 
                                                                                datetime_format = '%Y-%m-%dT%H:%M:%S+00:00'))


# # Get date from Date time:

# In[59]:


heights_tide_df['date'] = heights_tide_df['datetime'].apply(lambda x: x.split("T")[0])
extremes_tide_df['date'] = extremes_tide_df['datetime'].apply(lambda x: x.split("T")[0])


# ## Create date_coordinates column which will be used as PK:

# In[60]:


heights_tide_df["date_beach_name"] = heights_tide_df[
                                                     ['date', 'beach_name']
                                                     ].apply(lambda x: '|'.join(x.astype(str).values), axis=1)
# heights_tide_df.head()


# In[61]:


extremes_tide_df["date_beach_name"] = extremes_tide_df[
                                                     ['date', 'beach_name']
                                                     ].apply(lambda x: '|'.join(x.astype(str).values), axis=1)
# extremes_tide_df.head()


# # CREATE TIDE_HEIGHT_24HR_TABLE:

# In[62]:


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

create_table_query = """ 
                            CREATE TABLE IF NOT EXISTS TIDE_HEIGHT_TABLE (
                                tide_height_id SERIAL PRIMARY KEY ,
                                date DATE,
                                beach_id INTEGER NOT NULL,
                                beach_name TEXT,
                                beach_latitude REAL,
                                beach_longitude REAL,
                                beach_state TEXT,
                                timestamp INTEGER,
                                datetime TEXT,
                                height REAL,
                                state TEXT,
                                origin_distance REAL,
                                origin_distance_unit TEXT,
                                origin_latitude REAL,
                                origin_longitude REAL,
                                timezone TEXT,
                                date_beach_name TEXT,
                                FOREIGN KEY (beach_id) REFERENCES beach_table(beach_id) ON DELETE CASCADE
                                ); 
                            """

cur.execute(create_table_query)
conn.commit()

cur.close()
conn.close()


# # INSERTION TIDE_HEIGHT_24HR_TABLE:

# In[63]:


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

fill_question_mark_str = str(tuple(["%s"  for i in heights_tide_df.columns.tolist()])).replace("'", "")
fill_question_mark_str

for row in heights_tide_df.itertuples():
    data_tuple = tuple(row[1:])

#     print("data_tuple = ", data_tuple)
#     print(" ")
    
    cur.execute("""
                        INSERT INTO TIDE_HEIGHT_TABLE
                        (
                                date,
                                beach_id,
                                beach_name,
                                beach_latitude,
                                beach_longitude,
                                beach_state,
                                timestamp,
                                datetime,
                                height,
                                state,
                                origin_distance,
                                origin_distance_unit,
                                origin_latitude,
                                origin_longitude,
                                timezone,
                                date_beach_name
                         ) VALUES  
                         """ + fill_question_mark_str + " ;"
                , data_tuple)    

conn.commit()
cur.close()
conn.close()


# # CREATE EXTREMES_HEIGHT_24HR_TABLE:

# In[64]:


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

create_table_query = """ 
                            CREATE TABLE IF NOT EXISTS EXTREMES_HEIGHT_TABLE (
                                extremes_height_id SERIAL PRIMARY KEY ,
                                date DATE,
                                beach_id INTEGER NOT NULL,
                                beach_name TEXT,
                                beach_latitude REAL,
                                beach_longitude REAL,
                                beach_state TEXT,
                                timestamp INTEGER,
                                datetime TEXT,
                                height REAL,
                                state TEXT,
                                date_beach_name TEXT,
                                FOREIGN KEY (beach_id) REFERENCES beach_table(beach_id) ON DELETE CASCADE
                                ); 
                            """

cur.execute(create_table_query)
conn.commit()
cur.close()
conn.close()


# # INSERTION EXTREMES_HEIGHT_24HR_TABLE:

# In[65]:


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

fill_question_mark_str = str(tuple(["%s"  for i in extremes_tide_df.columns.tolist()])).replace("'", "")
fill_question_mark_str

for row in extremes_tide_df.itertuples():
    data_tuple = tuple(row[1:])

#     print("data_tuple = ", data_tuple)
#     print(" ")
    
    cur.execute("""
                        INSERT INTO EXTREMES_HEIGHT_TABLE
                        (
                            date,
                            beach_id,
                            beach_name,
                            beach_latitude,
                            beach_longitude,
                            beach_state,
                            timestamp,
                            datetime,
                            height,
                            state,
                            date_beach_name
                         ) VALUES  
                         """ + fill_question_mark_str + " ;"
                , data_tuple)    

conn.commit()
cur.close()
conn.close()


# # AGGREGATES TIDES:

# In[66]:


req_beach_loc_attri_lst = [ 
                            "date",
                            "beach_id",
                            "beach_name",
                            "beach_latitude",
                            "beach_longitude",
                            "beach_state",
                            "date_beach_name",
                            ]


# ## HEIGHTS FALL TIDES AGG:

# In[67]:


agg_ops_lst = ['max','min','mean','median','std','var','sem']
numeric_features = ["height"]
agg_dict = dict(zip(numeric_features, [agg_ops_lst for i in range(len(numeric_features))]))
# agg_dict

agg_fall_heights_tide_df = heights_tide_df.loc[heights_tide_df['state'] == "FALLING",:
                                              ].groupby([
                                                        'date_beach_name',
                                                        ]).agg(agg_dict)


agg_fall_heights_tide_df.columns = ["_fall_".join(x) for x in agg_fall_heights_tide_df.columns.ravel()]

agg_fall_heights_tide_df =  pd.merge(
                               agg_fall_heights_tide_df, 
                               heights_tide_df[req_beach_loc_attri_lst],
                               left_on = "date_beach_name",
                               right_on = "date_beach_name",
                               how = "inner"
                                    )

agg_fall_heights_tide_df.drop_duplicates(inplace = True)

agg_fall_heights_tide_df.reset_index(inplace = True, 
                                     drop = True)

print("agg_fall_heights_tide_df.shape = ", agg_fall_heights_tide_df.shape)
# agg_fall_heights_tide_df.head()


# In[68]:


# log_dataset_path = "D:\Monash_University_Stuff\Final_Semester\IE\Surfers_Bible_Code_Commit\Datasets\Daily_weather_tide_log\\"
# agg_fall_heights_tide_df.to_csv(log_dataset_path + "agg_fall_heights_tide_df"+"_10_05"+".csv")


# ## HEIGHTS RISE TIDES AGG:

# In[69]:


agg_ops_lst = ['max','min','mean','median','std','var','sem']
numeric_features = ["height"]
agg_dict = dict(zip(numeric_features, [agg_ops_lst for i in range(len(numeric_features))]))
# agg_dict

agg_rise_heights_tide_df = heights_tide_df.loc[heights_tide_df['state'] == "RISING",:
                                              ].groupby([
                                                        'date_beach_name',
                                                        ]
                                                       ).agg(agg_dict)

agg_rise_heights_tide_df.columns = ["_rise_".join(x) for x in agg_rise_heights_tide_df.columns.ravel()]

agg_rise_heights_tide_df =  pd.merge(
                               agg_rise_heights_tide_df, 
                               heights_tide_df[req_beach_loc_attri_lst],
                               left_on = "date_beach_name",
                               right_on = "date_beach_name",
                               how = "inner"
                                    )

agg_rise_heights_tide_df.drop_duplicates(inplace = True)

agg_rise_heights_tide_df.reset_index(inplace = True, 
                                     drop = True)

print("agg_rise_heights_tide_df.shape = ", agg_rise_heights_tide_df.shape)
# agg_rise_heights_tide_df.head()


# In[70]:


# log_dataset_path = "D:\Monash_University_Stuff\Final_Semester\IE\Surfers_Bible_Code_Commit\Datasets\Daily_weather_tide_log\\"
# agg_rise_heights_tide_df.to_csv(log_dataset_path + "agg_rise_heights_tide_df"+"_10_05"+".csv")


# ## EXTREMES HIGH TIDES AGG:

# In[71]:


agg_ops_lst = ['max','min','mean']
numeric_features = ["height"]
agg_dict = dict(zip(numeric_features, [agg_ops_lst for i in range(len(numeric_features))]))
# agg_dict

agg_high_tide_extremes_tide_df = extremes_tide_df.loc[extremes_tide_df['state'] == "HIGH TIDE",:
                                              ].groupby([
                                                        'date_beach_name',
                                                        ]).agg(agg_dict)

agg_high_tide_extremes_tide_df.columns = ["_high_tide_".join(x) for x in agg_high_tide_extremes_tide_df.columns.ravel()]

agg_high_tide_extremes_tide_df =  pd.merge(
                               agg_high_tide_extremes_tide_df, 
                               extremes_tide_df[req_beach_loc_attri_lst],
                               left_on = "date_beach_name",
                               right_on = "date_beach_name",
                               how = "inner"
                                    )

agg_high_tide_extremes_tide_df.drop_duplicates(inplace = True)

agg_high_tide_extremes_tide_df.reset_index(inplace = True, 
                                     drop = True)

print("agg_high_tide_extremes_tide_df.shape = ", agg_high_tide_extremes_tide_df.shape)
# agg_high_tide_extremes_tide_df.head()


# In[72]:


# log_dataset_path = "D:\Monash_University_Stuff\Final_Semester\IE\Surfers_Bible_Code_Commit\Datasets\Daily_weather_tide_log\\"
# agg_high_tide_extremes_tide_df.to_csv(log_dataset_path + "agg_high_tide_extremes_tide_df"+"_10_05"+".csv")


# ## EXTREMES LOW TIDES AGG:

# In[73]:


agg_ops_lst = ['max','min','mean']
numeric_features = ["height"]
agg_dict = dict(zip(numeric_features, [agg_ops_lst for i in range(len(numeric_features))]))
# agg_dict

agg_low_tide_extremes_tide_df = extremes_tide_df.loc[extremes_tide_df['state'] == "LOW TIDE",:
                                              ].groupby(['date_beach_name']).agg(agg_dict)

agg_low_tide_extremes_tide_df.columns = ["_low_tide_".join(x) for x in agg_low_tide_extremes_tide_df.columns.ravel()]

agg_low_tide_extremes_tide_df =  pd.merge(
                               agg_low_tide_extremes_tide_df, 
                               extremes_tide_df[req_beach_loc_attri_lst],
                               left_on = "date_beach_name",
                               right_on = "date_beach_name",
                               how = "inner"
                                    )

agg_low_tide_extremes_tide_df.drop_duplicates(inplace = True)

agg_low_tide_extremes_tide_df.reset_index(inplace = True, 
                                     drop = True)

print("agg_low_tide_extremes_tide_df.shape = ", agg_low_tide_extremes_tide_df.shape)
# agg_low_tide_extremes_tide_df.head()


# In[74]:


# log_dataset_path = "D:\Monash_University_Stuff\Final_Semester\IE\Surfers_Bible_Code_Commit\Datasets\Daily_weather_tide_log\\"
# agg_low_tide_extremes_tide_df.to_csv(log_dataset_path + "agg_low_tide_extremes_tide_df"+"_10_05"+".csv")


# # END OF TIDE API PROCESSING:
# 
# ---
# ---
# ---

# # Joing WEATHER AND TIDE AGG DF:

# In[75]:


# agg_weather_df.head()


# In[76]:


# agg_rise_heights_tide_df.head()


# In[79]:


tide_df_lst = [
                agg_rise_heights_tide_df, agg_fall_heights_tide_df,
                agg_high_tide_extremes_tide_df, agg_low_tide_extremes_tide_df,
#                 agg_datum_tide_df,
              ]
join_weather_tide_df = agg_weather_df.copy()
for df in tide_df_lst:
    join_weather_tide_df = pd.merge(
                               join_weather_tide_df, 
                               df.drop(['date', 'beach_name', 'beach_name', 'beach_state', 
                                        'beach_latitude','beach_longitude'], axis = 1, inplace = False),
                               left_on = "date_beach_name",
                               right_on = "date_beach_name",
                               how = "inner")
# drop_col_lst = []
# for col in join_weather_tide_df.columns.tolist():
#     if col.endswith("_x")|col.endswith("_y"):
#         drop_col_lst.append(col)
        
# join_weather_tide_df.drop(drop_col_lst, 
#                           axis=1, 
#                           inplace=True)

        
# join_weather_tide_df.head()


# In[80]:


# join_weather_tide_df.shape


# In[81]:


def get_month_season_month_day_func(date):
    month = date.split("-")[1]
    month_day = date.split("-")[2]
    if (int(month) == 9) | (int(month) == 10) | (int(month) == 11):
        season = "spring"
    elif (int(month) == 12) | (int(month) == 1) | (int(month) == 2):
        season = "summer"
    elif (int(month) == 3) | (int(month) == 4) | (int(month) == 5):
        season = "autumn"
    elif (int(month) == 6) | (int(month) == 7) | (int(month) == 8):
        season = "winter"
    return month, season, month_day
join_weather_tide_df['month'], join_weather_tide_df['season'], join_weather_tide_df['month_day'] = zip(*join_weather_tide_df.apply(lambda x: 
                                                                                        get_month_season_month_day_func(
                                                                                            x['date']), 
                                                                                        axis = 1))
# join_weather_tide_df.head()


# # ML MODEL INFERENCE:

# In[82]:


numeric_features_lst = [
#     Using Weather features
    'temperature_max',
 'temperature_min',
 'temperature_mean',
 'temperature_median',
 'temperature_std',
 'temperature_var',
 'temperature_sem',
 'apparentTemperature_max',
 'apparentTemperature_min',
 'apparentTemperature_mean',
 'apparentTemperature_median',
 'apparentTemperature_std',
 'apparentTemperature_var',
 'apparentTemperature_sem',
 'dewPoint_max',
 'dewPoint_min',
 'dewPoint_mean',
 'dewPoint_median',
 'dewPoint_std',
 'dewPoint_var',
 'dewPoint_sem',
 'humidity_max',
 'humidity_min',
 'humidity_mean',
 'humidity_median',
 'humidity_std',
 'humidity_var',
 'humidity_sem',
 'windSpeed_max',
 'windSpeed_min',
 'windSpeed_mean',
 'windSpeed_median',
 'windSpeed_std',
 'windSpeed_var',
 'windSpeed_sem',
 'windBearing_max',
 'windBearing_min',
 'windBearing_mean',
 'windBearing_median',
 'windBearing_std',
 'windBearing_var',
 'windBearing_sem',
 'uvIndex_max',
 'uvIndex_min',
 'uvIndex_mean',
 'uvIndex_median',
 'uvIndex_std',
 'uvIndex_var',
 'uvIndex_sem',
 'cloudCover_max',
 'cloudCover_min',
 'cloudCover_mean',
 'cloudCover_median',
 'cloudCover_std',
 'cloudCover_var',
 'cloudCover_sem',
# Tides Features                        
 'height_fall_max',
 'height_fall_min',
 'height_fall_mean',
 'height_fall_median',
 'height_fall_std',
 'height_fall_var',
 'height_fall_sem',
 'height_rise_max',
 'height_rise_min',
 'height_rise_mean',
 'height_rise_median',
 'height_rise_std',
 'height_rise_var',
 'height_rise_sem',
 'height_high_tide_max',
 'height_high_tide_min',
 'height_high_tide_mean',
 'height_low_tide_max',
 'height_low_tide_min',
 'height_low_tide_mean',
#  'LAT_datum_mean',
#  'HAT_datum_mean',
 'beach_latitude',
 'beach_longitude'
]
cat_features_lst = [
 'month',
 'month_day',
 'season']


# # SHARK ATTACK PREDICTION:

# In[83]:


params = {'loss_function':'Logloss', # objective function
          'eval_metric':'Accuracy', # metric
          'verbose': 200, # output to stdout info about training process every 200 iterations
          'random_seed': 100
         }

model = CatBoostClassifier(**params)
model.load_model('catboost_model_rand_search_tide_weather_shark_feat')


# In[84]:


join_weather_tide_df["shark_attack_percentage"] = 0.0
for index, row in join_weather_tide_df[numeric_features_lst+cat_features_lst].iterrows():
#     print(model.predict_proba(row.values)[1])
    join_weather_tide_df.loc[index, "shark_attack_percentage"] = round(float(model.predict_proba(row.values)[1])*100.0, 3)
#     join_weather_tide_df.loc[index, "shark_attack_percentage"] = model.predict_proba(row.values)[1]

# join_weather_tide_df.head(5)


# # SHARK SIGHTING PREDICTION:

# # Load CSV clean addresses with Coordinates CSV as Dataframe:

# In[121]:


# dataset_path = r"D:\Monash_University_Stuff\Final_Semester\IE\Surfers_Bible_Code_Commit\Datasets\\"
dataset_path = r""
join_shark_weather_tide_df = pd.read_csv("for_ml_join_shark_weather_tide_df.csv")
# raw_shark_df = pd.read_excel(dataset_path + "shark_file_geolocation_checkpoint.xlsx")
print("join_shark_weather_tide_df.shape = ", join_shark_weather_tide_df.shape)
join_shark_weather_tide_df.rename(columns = {
                                            "lat":"beach_latitude",
                                            "lng":"beach_longitude"}, inplace = True)
# join_shark_weather_tide_df.tail(5)


# # Using Cosine Similarity: Comparing point with distribution

# In[124]:


# %%time
# Wall time: 2min 36s
threshold = 0.90
join_weather_tide_df['shark_sighting_percentage'] = 0.0
for index_o, row_o in join_weather_tide_df[numeric_features_lst].iterrows():
    cos_sim_lst = []
    for index_i, row_i in join_shark_weather_tide_df[numeric_features_lst].iterrows():
    
        cosine_similarity_score = cosine_similarity(
                                                    X = row_o.values.reshape(1, -1), 
                                                    Y = row_i.values.reshape(1, -1), 
                                                    dense_output=True).flatten()[0]
        
        cos_sim_lst.append(cosine_similarity_score)
    
    no_vals_above = len([i for i in cos_sim_lst if i > threshold])
    
    percent_vals_above = round((no_vals_above/float(join_shark_weather_tide_df.shape[0]))*100.0, 3)
    
    join_weather_tide_df.loc[index_o, 'shark_sighting_percentage'] = percent_vals_above


# In[125]:


def percent_2_level_func(percent):
    if (percent > 0) & (percent <= 25):
        return "Low"
    elif (percent > 25) & (percent <= 50):
        return "Moderately Low"
    elif (percent > 50) & (percent <= 75):
        return "Moderately High"
    elif (percent > 75) & (percent <= 100):
        return "High"


# In[126]:


join_weather_tide_df["shark_sighting_level"] = join_weather_tide_df["shark_sighting_percentage"].apply(percent_2_level_func)
join_weather_tide_df["shark_attack_level"] = join_weather_tide_df["shark_attack_percentage"].apply(percent_2_level_func)
# join_weather_tide_df.head(10)


# In[127]:


# log_dataset_path = "D:\Monash_University_Stuff\Final_Semester\IE\Surfers_Bible_Code_Commit\Datasets\Daily_weather_tide_log\\"
# join_weather_tide_df.to_csv(log_dataset_path + "join_weather_tide_df"+"_10_05"+".csv")


# In[128]:


# join_weather_tide_df
join_weather_tide_df = join_weather_tide_df.loc[:,~join_weather_tide_df.columns.duplicated()]
join_weather_tide_df.drop(['beach_id_y'], axis=1, inplace=True)
join_weather_tide_df.rename(columns = {"beach_id_x": "beach_id"}, inplace =True)


# # CREATE join_weather_tide_TABLE:

# In[129]:


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

create_table_query = """ 
                            CREATE TABLE IF NOT EXISTS SHARK_PREDICTION_TABLE (
                                shark_prediction_id SERIAL PRIMARY KEY ,
                                date_beach_name TEXT,
                                temperature_max REAL,
                                temperature_min REAL,
                                temperature_mean REAL,
                                temperature_median REAL,
                                temperature_std REAL,
                                temperature_var REAL,
                                temperature_sem REAL,
                                apparentTemperature_max REAL,
                                apparentTemperature_min REAL,
                                apparentTemperature_mean REAL,
                                apparentTemperature_median REAL,
                                apparentTemperature_std REAL,
                                apparentTemperature_var REAL,
                                apparentTemperature_sem REAL,
                                dewPoint_max REAL,
                                dewPoint_min REAL,
                                dewPoint_mean REAL,
                                dewPoint_median REAL,
                                dewPoint_std REAL,
                                dewPoint_var REAL,
                                dewPoint_sem REAL,
                                humidity_max REAL,
                                humidity_min REAL,
                                humidity_mean REAL,
                                humidity_median REAL,
                                humidity_std REAL,
                                humidity_var REAL,
                                humidity_sem REAL,
                                windSpeed_max REAL,
                                windSpeed_min REAL,
                                windSpeed_mean REAL,
                                windSpeed_median REAL,
                                windSpeed_std REAL, 
                                windSpeed_var REAL,
                                windSpeed_sem REAL,
                                windBearing_max REAL,
                                windBearing_min REAL,
                                windBearing_mean REAL,
                                windBearing_median REAL,
                                windBearing_std REAL,
                                windBearing_var REAL,
                                windBearing_sem REAL,
                                uvIndex_max REAL,
                                uvIndex_min REAL,
                                uvIndex_mean REAL,
                                uvIndex_median REAL,
                                uvIndex_std REAL,
                                uvIndex_var REAL,
                                uvIndex_sem REAL,
                                cloudCover_max REAL,
                                cloudCover_min REAL,
                                cloudCover_mean REAL,
                                cloudCover_median REAL,
                                cloudCover_std REAL,
                                cloudCover_var REAL,
                                cloudCover_sem REAL,
                                date DATE,
                                beach_name TEXT,
                                beach_latitude REAL,
                                beach_longitude REAL,
                                beach_state TEXT,
                                height_rise_max REAL,
                                height_rise_min REAL,
                                height_rise_mean REAL,
                                height_rise_median REAL,
                                height_rise_std REAL,
                                height_rise_var REAL,
                                height_rise_sem REAL,
                                beach_id INTEGER,
                                height_fall_max REAL,
                                height_fall_min REAL,
                                height_fall_mean REAL,
                                height_fall_median REAL,
                                height_fall_std REAL,
                                height_fall_var REAL,
                                height_fall_sem REAL,
                                height_high_tide_max REAL,
                                height_high_tide_min REAL,
                                height_high_tide_mean REAL,
                                height_low_tide_max REAL,
                                height_low_tide_min REAL,
                                height_low_tide_mean REAL,
                                month TEXT,
                                season TEXT,
                                month_day TEXT,
                                shark_attack_percentage REAL,
                                shark_sighting_percentage REAL,
                                shark_sighting_level TEXT,
                                shark_attack_level TEXT,
                                FOREIGN KEY (beach_id) REFERENCES beach_table(beach_id) ON DELETE CASCADE
                                ); 
                            """

cur.execute(create_table_query)
conn.commit()

cur.close()
conn.close()


# # INSERTION join_weather_tide_TABLE:

# In[130]:


# %%time
# Wall time: 34 s
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

fill_question_mark_str = str(tuple(["%s"  for i in join_weather_tide_df.columns.tolist()])).replace("'", "")
fill_question_mark_str

for row in join_weather_tide_df.itertuples():
    data_tuple = tuple(row[1:])

    print("data_tuple = ", data_tuple)
    print(" ")
    
    cur.execute("""
                        INSERT INTO SHARK_PREDICTION_TABLE
                        (
                            date_beach_name,
                            temperature_max,
                            temperature_min,
                            temperature_mean,
                            temperature_median,
                            temperature_std,
                            temperature_var,
                            temperature_sem,
                            apparentTemperature_max,
                            apparentTemperature_min,
                            apparentTemperature_mean,
                            apparentTemperature_median,
                            apparentTemperature_std,
                            apparentTemperature_var,
                            apparentTemperature_sem,
                            dewPoint_max,
                            dewPoint_min,
                            dewPoint_mean,
                            dewPoint_median,
                            dewPoint_std,
                            dewPoint_var,
                            dewPoint_sem,
                            humidity_max,
                            humidity_min,
                            humidity_mean,
                            humidity_median,
                            humidity_std,
                            humidity_var,
                            humidity_sem,
                            windSpeed_max,
                            windSpeed_min,
                            windSpeed_mean,
                            windSpeed_median,
                            windSpeed_std,
                            windSpeed_var,
                            windSpeed_sem,
                            windBearing_max,
                            windBearing_min,
                            windBearing_mean,
                            windBearing_median,
                            windBearing_std,
                            windBearing_var,
                            windBearing_sem,
                            uvIndex_max,
                            uvIndex_min,
                            uvIndex_mean,
                            uvIndex_median,
                            uvIndex_std,
                            uvIndex_var,
                            uvIndex_sem,
                            cloudCover_max,
                            cloudCover_min,
                            cloudCover_mean,
                            cloudCover_median,
                            cloudCover_std,
                            cloudCover_var,
                            cloudCover_sem,
                            date,
                            beach_name,
                            beach_latitude,
                            beach_longitude,
                            beach_state,
                            height_rise_max,
                            height_rise_min,
                            height_rise_mean,
                            height_rise_median,
                            height_rise_std,
                            height_rise_var,
                            height_rise_sem,
                            beach_id,
                            height_fall_max,
                            height_fall_min,
                            height_fall_mean,
                            height_fall_median,
                            height_fall_std,
                            height_fall_var,
                            height_fall_sem,
                            height_high_tide_max,
                            height_high_tide_min,
                            height_high_tide_mean,
                            height_low_tide_max,
                            height_low_tide_min,
                            height_low_tide_mean,
                            month,
                            season,
                            month_day,
                            shark_attack_percentage,
                            shark_sighting_percentage,
                            shark_sighting_level,
                            shark_attack_level
                         ) VALUES  
                         """ + fill_question_mark_str + " ;"
                , data_tuple)    

conn.commit()

cur.close()
conn.close()


# ---
# ---
# ---

# In[ ]:




