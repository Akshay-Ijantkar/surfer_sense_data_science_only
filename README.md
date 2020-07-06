# Surfers' Sense - One Stop Solution Prior Surfing

Surfers’ Sense is mobile compatible web application which helps novice surfer to plan safe surf journey along with support services which will keep them updated about what’s happening in surfing world and help them in the emergency situation. The application was designed and built keeping in mind surfer at rudimentary level, but it can be useful for surfers and swimmers of all experienced levels. 

It provides following functionalities: 
* 3 days forecast for all Victorian beaches of Shark Attack Probabilities using Machine Learning (ML) model. Model trained on historic shark attack data combining it with aggregated historic weather and tides data for that particular location and date where shark attack incidents have happened. 
* 3 days forecast for all Victorian beaches of Shark Sighting Probabilities using Statistical Model wherein it calculates similarity score of the given day by comparing aggregated weather and tide data of the date and location which you intend to calculate with aggregated historic weather and tides data where shark attack has happened. 
* Suggesting necessary surf gears for surfing according to weather conditions like sea water temperature. 
* Simplified Surf report which includes wave period, wave height, tides status and sunrise sunset timings. 
* Personalized Surfboard specifications recommendation using physical parameters like age, weight, fitness, etc. along with surfing skill levels. 
* Tailored surfing news related to topics which surfers will be interested into.  o Topics which will give surfers word of cautions like Shark Attack, Jelly fish and Rip Current.  o Topics which will motivate surfers like Surfing competitions and Surfing life. 
* Emergency support service which will help surfers in case of emergency like injuries to navigate with nearby hospitals as per their current location. It just not recommends nearby hospitals with generic information like address, website, phone no. and google rating but also tells user responsiveness score (Percentage of patients seen on time by that hospital in the past) and also gives list of services provided by the hospital. 

# Product Link https://surfersense.tk/   
* Access Credentials: Username: user | Password: EnterSite123@ 

# Product Video: Access Link: https://www.youtube.com/watch?v=1cWaSNg8Heg

# How Surfers' Sense Works from Data Science aspect - Explained in Brief:
* Video Access Link: https://drive.google.com/file/d/1C3vfXbtRLm7Cf19w8wzFhZXiDY3ctOR0/view?usp=sharing 

# DB_Populating_Scripts:
-> Running on AWS EC2 to populate Postgres DB everynight using Crontab

## Daily_Scheduler_Weather_Tide_API.py

* Requests 24hour log data of weather everyday of the 4th day from current day and inserts in Postgres DB
* Calculate aggregate of all weather attributes over 24 hour
* Requests 24hour log data of tides everyday of the 4th day from current day and inserts in Postgres DB
* Calculate aggregate of all tide status attributes over 24 hour
* Calculate aggregate of all extremes todes attributes over 24 hour
* This aggregated weather and tides will be features will be input for the trained Catboost Model
* Model (catboost_model_rand_search_tide_weather_shark_feat) is infered using above weather and tide features and SHARK ATTACK PROBABILITY will be output
* SHARK ATTACK PROBABILITY is converted to SHARK ATTACK LEVELS to make user readeable
* SHARK SIGHTING PROBABILITY is calculated cosine similarity and algorithm discussed (for_ml_join_shark_weather_tide_df.csv) in Maintenance DOC
* SHARK SIGHTING PROBABILITY is converted to SHARK SIGHTING LEVELS to make user readeable
* Inserts Combined aggregated weather, tide and Model Predictions (SHARK ATTACK PROBABILITY and SHARK SIGHTING PROBABILITY) DF into Postgres DB 

## Daily_Scheduler_Swell_Pollution_Astro_Update.py

* Requests 24hour log data of SWELLS everyday of the 4th day from current day
* SWELLS (Wave period, Height & Direction): Response JSON dictionary is parsed and converted in to DF rows and inserted into Postgres DB
* Requests 24hour log data of ASTRONOMY everyday of the 4th day from current day
* ASTRONOMY (Sunrise and Sunset Time): Response JSON dictionary is parsed and converted in to DF rows and inserted into Postgres DB
* Requests 24hour log data of SEA WATER QUALITY everyday of the 4th day from current day
* SEA WATER QUALITY (ph): Response JSON dictionary is parsed and converted in to DF rows and inserted into Postgres DB

## Surfing_News_Scheduler.py
* Requests news related to topics:Shark Attack, Jelly Fish Incidents, Rip Current Incidents, Surfing News Australia, Surfing Competition News
* Filters the news using Keyword search to remove off topic news to improve new relevance score.
* Parses Title, Published Date, Description, Link, Images and inserts in Postgres DB

## Deleting_Rows_From_Tables_Scheduler.py

* Remove all records from dynamic tables(Refer Maintenance Doc for understanding Dyanamic Tables) of the last 5th day from current day so that we will have records of last 5 days and forcast of 3 days so total of 8 days in the Postgres DB.
* I did above thing so that	I am using AWS RDS within free limit of data storage and save space.


# API_Parsing_Scripts:
## > Simplify_Swell_AND_Seawater_Quality.ipynb

* Convert numeric values into user readable values using below logic

* Required Gears : Depending on water temperature on a given date and location, we are suggesting what gears you should be wearing to have safe surfing experience and avoid sickness like Hypothermia.

Full Suit + Boots + Gloves + Hood -> Water Temperature below 58 degree °F
Full Suit + Boots -> Water Temperature between 58 °F and 63 °F
Full Suit -> Water Temperature between 63 °F and 68 °F
Top + Shorts -> Water Temperature between 68 °F and 75 °F
Rashguard Enough -> Water Temperature above 75 °F



* Wave Period: The amount of time it takes for two successive wave crests to pass through a determined point, it measures the quality of the upcoming surf session.

Un-surfable -> Wave period less than 5 minutes
Weak -> Wave period between 5 to 8 minutes
Average -> Wave period between 8 to 10 minutes
Good -> Wave period between 10 to 12 minutes
Excellent -> Wave period above 12 minutes



* Wave Height: The height of unbroken waves as they approach the beach in deep water.

Ankle to Knee -> Wave Height between 0 ft to 1 ft
Knee to Thigh -> Wave Height between 1 ft to 2 ft
Thigh to Waist -> Wave Height between 2 ft to 3 ft
Waist to Chest -> Wave Height between 3 ft to 4 ft
Chest to Head -> Wave Height between 4 ft to 5 ft
Above Head -> Wave Height above 5 ft


## > Daily_Scheduler_Swell_Pollution_Astro_Update.ipynb
* Notebook of Daily_Scheduler_Swell_Pollution_Astro_Update.py script

## > Daily_Scheduler_Weather_Tide_API.ipynb
* Notebook of Daily_Scheduler_Weather_Tide_API.py script

## > Deleting_Rows_From_Tables_Scheduler.ipynb
* Notebook of Deleting_Rows_From_Tables_Scheduler.py script

## > HOSPITAL_DATA_ANALYSIS.ipynb
* Get 200 hospitals from Victoria
* Select only required columns
* Select the columns which has percentage of patients seen on time in urgent cases as column

## > HOSPITAL_DETAILS_SERVICES.ipynb
* Using Google Places API get more details about all 200 hospitals like phone no., address, google ratings, opening hours, etc. 
* Get sevices provide by all hospitals as table and insert in Services table in PB
* Create and insert bridging table joining Hospital Table and Services table

## > Surfboard_Recommendations.ipynb
* Using user's attributes like Age, weight, Skills, Fitness, etc. Calculate Volume
* Using user's attributes like Age, weight, Skills, Fitness, etc. Calculate Dimensions

## > Surfing_News_Scheduler.ipynb
* Notebook of Surfing_News_Scheduler.py script

# Cleaning Scripts:

## Geolocation_Data_Wrangling_Shark_File.ipynb
* Cleaning Date from Shark Attack Incident File
* Cleaning Location from Shark Attack Incident File
* Cleaning Time from Shark Attack Incident File
* From the clean location text address get the Coodinates of that location using Google Geolocation API
* Required output will be Location Coordinates and Date

## Weather_Data_Wrangling_Shark_File_DARK_SKY.ipynb
* Once I have Date and Location of all Shark Attack Incident in Australia from the above file
* Using DarkSky API I will request for 24hour log weather attributes of that particular date and location where shark incident has happened
* Calculate all aggregates (Mean, STD, Median, SE, etc.) of all weather attributes.


## Tide_Extremes_Datums_API_2_Dataset_Cleaning.ipynb
* Once I have Date and Location of all Shark Attack Incident in Australia from the above file
* Using Tides Hood API I will request for 24hour log tides and extreme tides attributes of that particular date and location where shark attack incidents has happened 
* Calculate all aggregates (Mean, STD, Median, SE, etc.) of all tides attributes.
* So Aggregated tide and weather attributes along with coordinates of all location where Shark Incident has happened in the history will be features for Machine Learning Model.

## Shark_Sighting_Calculate_Mechanism.ipynb
* Comparing aggregated weather and tides features from historic shark incident data with current aggregated weather and tides features on particular date and beach
* Tried different distance metrics like cosine, Mahattan distance, Euclidean, etc. 
* Finally I choosed cosine similarity since it has fixed range from 0 to 1 and it will be easy to interpret and can be make it easy to read by end users.