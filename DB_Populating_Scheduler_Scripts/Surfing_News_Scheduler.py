#!/usr/bin/env python
# coding: utf-8

# ### Author: Akshay Ijantkar
# ### Team: Aqua Wizards
# ### Project: Surfers Bible

# * https://launchschool.com/books/sql/read/table_relationships

# # Import Libraries:

# 0 1 * * * /usr/bin/python3 /home/ubuntu/pop_db_sch_ss/Daily_Scheduler_Swell_Pollution_Astro_News_API.py >> /home/ubuntu/pop_db_sch_ss/log_Daily_Scheduler_Swell_Pollution_Astro_News_API.txt 2>&1

# In[1]:


# import nltk

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
rcParams['figure.figsize'] = 50,50
import pandas_profiling
# pd.set_option('display.max_rows', 1500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
from pandas import ExcelWriter
from pandas import ExcelFile

from pygeocoder import Geocoder

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

# In[2]:


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


# # Everything Endpoint:

# In[3]:


news_col_lst = ['date','news_topic','source', 'author', 'title', 'description', 'url', 'urlToImage', 'publishedAt']

news_df = pd.DataFrame(columns = news_col_lst)
news_df


# In[4]:


import re
def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


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

# cur.execute("DROP TABLE SEA_WATER_QUALITY_TABLE;")


NEWS_TABLE_df = pd.read_sql("SELECT *  FROM NEWS_TABLE;", conn)

cur.close()
conn.close()
# NEWS_TABLE_df.head()


# In[6]:


topics_lst = [
'shark+australia',
'beach+jellyfish+australia',
'surfing+events',
'beach+rip+currents',
'beach+surfing+australia',
'surfing+guide',
]

topics_keywords_lst = [
"shark+australia", 
"beach+jellyfish+australia", 
"surfing+events+competition+Margaret+River+Pro+Rip+Curl+Pro+Australian+Surf+Life+Saving+Championships+Surfest+Quiksilver+Pro",
"beach+rip+currents+ripcurrents+drowning", 
"beach+surfing+life+australia+weather+waves", 
"surfingaustralia+guide+tutorial",
]
for topics_keywords, topic in zip(topics_keywords_lst[:], topics_lst[:]): 
    news_row_dict = {}
    
    q_str = qInTitle_str = topics_keywords
    NEWS_API_KEY = ""

    from_to_date = today_date

    get_request_url_str = "https://newsapi.org/v2/everything?"

    get_request_url_str += "q="
    get_request_url_str += q_str

    get_request_url_str += "&qInTitle"
    get_request_url_str += qInTitle_str

    get_request_url_str += "&sortBy="
    get_request_url_str += "relevancy"

    get_request_url_str += "&from"
    get_request_url_str += from_to_date


    get_request_url_str += "&to"
    get_request_url_str += from_to_date

    get_request_url_str += "&language"
    get_request_url_str += "en"

    get_request_url_str += "&country"
    get_request_url_str += "au"

    get_request_url_str += "&apiKey="
    get_request_url_str += NEWS_API_KEY
    

    try:
        response_dict = json.loads(requests.get(get_request_url_str).text)

        if response_dict['status'] == 'ok':

            if response_dict['totalResults'] > 0:
#                 print("len response_dict['articles'] = ", len(response_dict['articles']))
#                 print("response_dict['articles'] = ", response_dict['articles'])


                for article_dict in response_dict['articles']:
                    if article_dict['title'] not in NEWS_TABLE_df.title.values.tolist():
                        news_row_dict = {}

                        news_row_dict['date'] = today_date
                        news_row_dict['news_topic'] = topic

                        news_row_dict['source'] = article_dict['source']['name']
                        news_row_dict['author'] = article_dict['author']
                        news_row_dict['title'] = article_dict['title']
                        news_row_dict['description'] = cleanhtml(article_dict['description']) 
                        news_row_dict['url'] = article_dict['url']
                        news_row_dict['urlToImage'] = article_dict['urlToImage']
                        news_row_dict['publishedAt'] = article_dict['publishedAt']

                        news_df = news_df.append(news_row_dict, ignore_index=True)

        else:
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("Status failed!!")
            print("topic = ", topic)
            print("date = ", given_date)
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    except:
        print("###############################################################################")
        print("Request failed !!!")
        print("topic = ", topic)
        print("date = ", given_date)
        print("###############################################################################")


# # Convert UTC to AEST TIME of PublishedAt column:

# In[7]:


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
    


# In[8]:


news_df["publishedAt"] = news_df['publishedAt'].apply(lambda x: convert_datetime_in_dif_timezones(
                                                                            from_datetime_str = x, 
                                                                            from_timezone_str = 'UTC', 
                                                                            to_timezone_str ='Australia/Melbourne', 
                                                                            datetime_format = '%Y-%m-%dT%H:%M:%SZ'))


# # CREATE NEWS_TABLE:

# In[9]:


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
      CREATE TABLE IF NOT EXISTS NEWS_TABLE
      (
        astronomy_id SERIAL PRIMARY KEY,
        date DATE,
        news_topic TEXT,
        source TEXT, 
        author TEXT, 
        title TEXT, 
        description TEXT, 
        url TEXT, 
        urlToImage TEXT, 
        publishedAt TEXT
       ); 
       '''

cur.execute(create_table_query)
conn.commit()

cur.close()
conn.close()


# # INSERT NEWS_TABLE:

# In[10]:


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

fill_question_mark_str = str(tuple(["%s"  for i in news_df.columns.tolist()])).replace("'", "")
fill_question_mark_str

for row in news_df.itertuples():
    data_tuple = tuple(row[1:])

#     print("data_tuple = ", data_tuple)
#     print(" ")
    
    cur.execute("""
                        INSERT INTO NEWS_TABLE
                        (
                        date,
                        news_topic,
                        source, 
                        author, 
                        title, 
                        description, 
                        url, 
                        urlToImage, 
                        publishedAt
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

# In[ ]:




