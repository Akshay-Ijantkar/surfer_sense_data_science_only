#!/usr/bin/env python
# coding: utf-8

# ### Author: Akshay Ijantkar
# ### Team: Aqua Wizards
# ### Project: Surfers Bible

# * https://launchschool.com/books/sql/read/table_relationships

# # Import Libraries:

# 0 1 * * * /usr/bin/python3 /home/ubuntu/pop_db_sch_ss/Daily_Scheduler_Swell_Pollution_Astro_News_API.py >> /home/ubuntu/pop_db_sch_ss/log_Daily_Scheduler_Swell_Pollution_Astro_News_API.txt 2>&1

# In[22]:


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
# rcParams['figure.figsize'] = 50,50
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

# In[23]:


from datetime import datetime
import datetime
no_days_from_today = -5

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


# # FUNCTIO  TO EXCUTE SQL QUERY:

# In[24]:


def execute_sql_func(sql_str, data_tuple):
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

    cur.execute(sql_str, data_tuple)

    conn.commit()

    cur.close()
    conn.close()


# # DELETE RECORDS FROM astronomy_table:

# In[54]:


sql_str = "DELETE FROM public.astronomy_table WHERE date = %s ;"
execute_sql_func(sql_str = sql_str, data_tuple = (given_date,))


# # DELETE RECORDS FROM extremes_height_table:

# In[25]:


sql_str = "DELETE FROM public.extremes_height_table WHERE date = %s ;"
execute_sql_func(sql_str = sql_str, data_tuple = (given_date,))


# # DELETE RECORDS FROM news_table:

# In[26]:


sql_str = "DELETE FROM public.news_table WHERE date = %s ;"
execute_sql_func(sql_str = sql_str, data_tuple = (given_date,))


# # DELETE RECORDS FROM sea_water_quality_table:

# In[27]:


sql_str = "DELETE FROM public.sea_water_quality_table WHERE date = %s ;"
execute_sql_func(sql_str = sql_str, data_tuple = (given_date,))


# # DELETE RECORDS FROM swell_table:

# In[28]:


sql_str = "DELETE FROM public.swell_table WHERE date = %s ;"
execute_sql_func(sql_str = sql_str, data_tuple = (given_date,))


# # DELETE RECORDS FROM tide_height_table:

# In[29]:


sql_str = "DELETE FROM public.tide_height_table WHERE date = %s ;"
execute_sql_func(sql_str = sql_str, data_tuple = (given_date,))


# # DELETE RECORDS FROM weather_table:

# In[30]:


sql_str = "DELETE FROM public.weather_table WHERE date = %s ;"
execute_sql_func(sql_str = sql_str, data_tuple = (given_date,))


# ---






