#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Question 1#CAR matrix generation
Under the function named generate_car_matrix write a logic that takes the dataset-1.csv as a DataFrame. Return a new DataFrame that follows the following rules:

values from id_2 as columns
values from id_1 as index
dataframe should have values from car column
diagonal values should be 0.


# In[17]:


import pandas as pd


# In[20]:


df = pd.read_csv("dataset-1.csv")


# In[25]:


import pandas as pd
# Read the CSV file into a DataFrame
df = pd.read_csv("dataset-1.csv")

# Pivot the DataFrame to create the desired matrix
matrix = df.pivot(index='id_1', columns='id_2', values='car').fillna(0)

# Set diagonal values to 0
for i in range(min(matrix.shape)):
    matrix.iloc[i, i] = 0

print(matrix) 


# In[4]:


Question 2: Car Type Count Calculation
Create a Python function named get_type_count that takes the dataset-1.csv as a DataFrame. Add a new categorical column car_type based on values of the column car:

low for values less than or equal to 15,
medium for values greater than 15 and less than or equal to 25,
high for values greater than 25.
Calculate the count of occurrences for each car_type category and return the result as a dictionary. Sort the dictionary alphabetically based on keys.


# In[5]:


Answer

import pandas as pd
def get_type_count(dataframe)
df=pd.read_csv(dataset-1.csv)
# adding a column car type based on car values
df['car type']=col
# conditions
df['car']<=15
df['car']>15 $df['car']<=25
df['car']>25

category = ['low', 'medium', 'high']
df['car_type'] = pd.Series(np.select(conditions, choices, default='Undefined'), index=df.index)

# Calculate count of occurrences for each 'car_type'
type_count = df['car_type'].value_counts().to_dict()

# Sort the dictionary alphabetically based on keys
type_count_sorted = dict(sorted(type_count.items()))

return type_count_sorted


# In[ ]:


Question 3: Bus Count Index Retrieval
Create a Python function named get_bus_indexes that takes the dataset-1.csv as a DataFrame. 
The function should identify and return the indices as a list (sorted in ascending order) where the bus values 
are greater than twice the mean value of the bus column in the DataFrame



# In[ ]:


import pandas as pd

def get_bus_indexes(df: pd.DataFrame) -> list:
bus_mean = df['bus'].mean()
bus_indexes = df[df['bus'] > 2 * bus_mean].index.tolist()
bus_indexes.sort()
return bus_indexes

#Assuming 'dataset_path' is the path to your CSV file
dataset_path = 'path_to_your_dataset-1.csv'

#Read the CSV file into a DataFrame
df = pd.read_csv(dataset_path)

#Call the function to get the bus indexes
result = get_bus_indexes(df)


# In[ ]:


Question 4: Route Filtering
Create a python function filter_routes that takes the dataset-1.csv as a DataFrame. 
The function should return the sorted list of values of column route for which 
the average of values of truck column is greater than 7.


# In[ ]:


import pandas as pd

def filter_routes(dataframe):
# Calculate the average value of the "truck" column for each route
route_avg_truck = dataframe.groupby('route')['truck'].mean()

# Filter routes where the average value of the "truck" column is greater than 7
filtered_routes = route_avg_truck[route_avg_truck > 7].index.tolist()

return filtered_routes
result = filter_routes(dataset)
print(result)


# In[ ]:


Question 5: Matrix Value Modification
Create a Python function named multiply_matrix that takes the resulting DataFrame from Question 1, as input and modifies each value according to the following logic:

If a value in the DataFrame is greater than 20, multiply those values by 0.75,
If a value is 20 or less, multiply those values by 1.25.
The function should return the modified DataFrame which has values rounded to 1 decimal place.



# In[ ]:


Answer
def modify_value(val):
if val > 20:
return round(val * 0.75, 1)
else:
return round(val * 1.25, 1)

df = df.applymap(modify_value)

return df


# In[ ]:


Question 6: Time Check
You are given a dataset, dataset-2.csv, containing columns id, id_2, and timestamp (startDay, startTime, endDay, endTime). 
The goal is to verify the completeness of the time data by checking whether the timestamps for 
each unique (id, id_2) pair cover a full 24-hour period (from 12:00:00 AM to 11:59:59 PM) and span 
all 7 days of the week (from Monday to Sunday).

Create a function that accepts dataset-2.csv as a DataFrame and returns a boolean series that indicates 
if each (id, id_2) pair has incorrect timestamps. The boolean series must have multi-index (id, id_2).


# In[ ]:


Answer
import pandas as pd
def check_time_completeness(dataframe):
# Combine date and time columns to create datetime objects
dataframe['start_datetime'] = pd.to_datetime(dataframe['startDay'] + ' ' + dataframe['startTime'])
dataframe['end_datetime'] = pd.to_datetime(dataframe['endDay'] + ' ' + dataframe['endTime'])
# Check if the time range for each ("id", "id_2") pair covers a full 24-hour period and spans all 7 days
completeness_series = dataframe.groupby(['id', 'id_2']).apply(lambda group: check_time_range(group)).droplevel(2)
return completeness_series
def check_time_range(group):
# Check if the time range covers a full 24-hour period and spans all 7 days
start_time = group['start_datetime'].min().time()
end_time = group['end_datetime'].max().time()
return (end_time > start_time) and (group['start_datetime'].dt.dayofweek.nunique() == 7)
result = check_time_completeness(dataset_2)
print(result)



# In[ ]:


Question 1: Distance Matrix Calculation
Create a function named calculate_distance_matrix that takes the dataset-3.csv as input and generates a DataFrame representing distances between IDs.

The resulting DataFrame should have cumulative distances along known routes, 
with diagonal values set to 0. If distances between toll locations A to B and B to C are known, then the distance
from A to C should be the sum of these distances. Ensure the matrix is symmetric, accounting for bidirectional 
distances between toll locations (i.e. A to B is equal to B to A).


# In[ ]:


import pandas as pd
def calculate_distance_matrix(df):
unique_ids = pd.unique(df[['id_start', 'id_end']].values.flatten())
distance_matrix = pd.DataFrame(index=unique_ids, columns=unique_ids, dtype=float)

distance_matrix.values[[range(len(unique_ids))]*2] = 0

for index, row in df.iterrows():
    id_start, id_end, distance = row['id_start'], row['id_end'], row['distance']
    distance_matrix.at[id_start, id_end] += distance
    distance_matrix.at[id_end, id_start] += distance

return distance_matrix


# In[ ]:


Question 2: Unroll Distance Matrix
Create a function unroll_distance_matrix that takes the DataFrame created in Question 1. The resulting DataFrame should have three columns: columns id_start, id_end, and distance.

All the combinations except for same id_start to id_end must be present in the rows with their distance values from the input DataFrame.


# In[ ]:


import pandas as pd

def unroll_distance_matrix(result_matrix: pd.DataFrame) -> pd.DataFrame:
result_matrix = result_matrix.reset_index()
unrolled_df = pd.melt(result_matrix, id_vars=['id_start'], var_name='id_end', value_name='distance')
return unrolled_df



# In[ ]:


Question 3: Finding IDs within Percentage Threshold
Create a function find_ids_within_ten_percentage_threshold that takes the DataFrame created in Question 2 and a reference value from the id_start column as an integer.

Calculate average distance for the reference value given as an input and return a sorted list of values from id_start column which lie within 10% (including ceiling and floor) of the reference value's average.


# In[ ]:


# Calculate the average distance for the reference value
average_distance = reference_df['distance'].mean()
# Calculate the lower and upper bounds within the 10% threshold
lower_bound = average_distance - (average_distance * 0.1)
upper_bound = average_distance + (average_distance * 0.1)
# Filter IDs within the 10% threshold
within_threshold_ids = distance_df[
    (distance_df['id_start'] != reference_value) &
    (distance_df['distance'] >= lower_bound) &
    (distance_df['distance'] <= upper_bound)]['id_start'].unique()
# Sort the list of IDs
sorted_within_threshold_ids = sorted(within_threshold_ids)
return sorted_within_threshold_ids


# In[ ]:


Question 4: Calculate Toll Rate
Create a function calculate_toll_rate that takes the DataFrame created in Question 2 as input and 
calculates toll rates based on vehicle types.

The resulting DataFrame should add 5 columns to the input DataFrame: moto, car, rv, bus, and truck with their respective rate coefficients. The toll rates should be calculated by multiplying the distance with the given rate coefficients for each vehicle type:

0.8 for moto
1.2 for car
1.5 for rv
2.2 for bus
3.6 for truck


# In[ ]:


import pandas as pd
def calculate_time_based_toll_rates(df: pd.DataFrame) -> pd.DataFrame:
# Assuming 'start_time' and 'end_time' columns are in datetime format
df['time_factor'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 3600

toll_rate_mapping = {
    'moto_rate': 0.1,
    'car_rate': 0.5,
    'rv_rate': 1.0,
    'bus_rate': 1.5,
    'truck_rate': 2.0
}

for vehicle_type, rate in toll_rate_mapping.items():
    df[f'discounted_{vehicle_type}'] = df[vehicle_type] * df['time_factor']

return df


# In[ ]:


Question 5: Calculate Time-Based Toll Rates
Create a function named calculate_time_based_toll_rates that takes the DataFrame created in Question 3 as input and calculates toll rates for different time intervals within a day.

The resulting DataFrame should have these five columns added to the input: start_day, start_time, end_day, and end_time.

start_day, end_day must be strings with day values (from Monday to Sunday in proper case)
start_time and end_time must be of type datetime.time() with the values from time range given below.
Modify the values of vehicle columns according to the following time ranges:

Weekdays (Monday - Friday):

From 00:00:00 to 10:00:00: Apply a discount factor of 0.8
From 10:00:00 to 18:00:00: Apply a discount factor of 1.2
From 18:00:00 to 23:59:59: Apply a discount factor of 0.8
Weekends (Saturday and Sunday):

Apply a constant discount factor of 0.7 for all times.
For each unique (id_start, id_end) pair, cover a full 24-hour period (from 12:00:00 AM to 11:59:59 PM) and span all 7 days of the week (from Monday to Sunday).


# In[ ]:


import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
def calculate_time_based_toll_rates(input_df):
# Convert start and end times to datetime objects
input_df['start_datetime'] = pd.to_datetime(input_df['startDay'] + ' ' + input_df['startTime'])
input_df['end_datetime'] = pd.to_datetime(input_df['endDay'] + ' ' + input_df['endTime'])
# Define time ranges and discount factors
time_ranges_weekdays = [(time(0, 0), time(10, 0), 0.8),
                        (time(10, 0), time(18, 0), 1.2),
                        (time(18, 0), time(23, 59, 59), 0.8)]
time_ranges_weekends = [(time(0, 0), time(23, 59, 59), 0.7)]
# Apply discount factors based on time ranges
for start_time, end_time, discount_factor in time_ranges_weekdays:
    mask = (input_df['start_datetime'].dt.time >= start_time) & (input_df['end_datetime'].dt.time <= end_time) & (input_df['start_datetime'].dt.dayofweek < 5)
    input_df.loc[mask, ['moto', 'car', 'rv', 'bus', 'truck']] *= discount_factor

for start_time, end_time, discount_factor in time_ranges_weekends:
    mask = (input_df['start_datetime'].dt.time >= start_time) & (input_df['end_datetime'].dt.time <= end_time)
    input_df.loc[mask, ['moto', 'car', 'rv', 'bus', 'truck']] *= discount_factor

# Add columns for start day, start time, end day, and end time
input_df['start_day'] = input_df['start_datetime'].dt.strftime('%A')
input_df['start_time'] = input_df['start_datetime'].dt.time
input_df['end_day'] = input_df['end_datetime'].dt.strftime('%A')
input_df['end_time'] = input_df['end_datetime'].dt.time

# Drop temporary columns
input_df = input_df.drop(['start_datetime', 'end_datetime'], axis=1)
return input_df
result_with_time_based_rates = calculate_time_based_toll_rates(unrolled_result)
print(result_with_time_based_rates)

