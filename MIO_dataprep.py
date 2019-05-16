import os
import pandas as pd
import numpy as np
from haversine import haversine

os.chdir("/Users/gabrielc/Dropbox (MIT)/MIT/Spring_2019/1263/SCM_course_project")

# Import Data
# df = pd.read_csv("data.csv")
df = pd.read_csv("cvs_option.csv")

# Filter Date
df['Date'] = (pd.to_datetime(df['DeliveryWindowA']).dt.date).astype(str)
df = df[df['Date'] == "2014-01-02"].reset_index()
df2 = df[['DeliveryWindowA', 'DeliveryWindowB', 'dropoff_y', 'dropoff_x']]

# Dropoff Nodes
df2['dropoff'] = list(zip(df2['dropoff_x'], df2['dropoff_y']))

dropoff_points = list(zip(df2['dropoff_x'], df2['dropoff_y']))
dropoff_points = [(-73.999744,40.758994)] + dropoff_points + [(-73.999743,40.758993)]

# Factorize the nodes 
nodes, node_dict = pd.factorize(dropoff_points)
np.append(node_dict, coord)
number_of_nodes = len(nodes)

demand = np.ones(number_of_nodes-2)
demand = np.insert(demand, 0, 0)
demand = np.append(demand, 0)


distance_matrix = np.zeros([number_of_nodes, number_of_nodes])

for i in range(0, number_of_nodes):
    for j in range(0, number_of_nodes):
        if i == j:
            distance_matrix[i,j] = 0
        else:
            distance_matrix[i,j] = haversine(node_dict[nodes[i]], node_dict[nodes[j]], unit='mi')

cost_matrix = distance_matrix * 5
cost_matrix = np.round(cost_matrix)

speed_matrix = np.random.normal(10, 2, (number_of_nodes, number_of_nodes))

time_matrix = distance_matrix/speed_matrix
time_matrix = np.round(time_matrix*60)

timing = df2[['DeliveryWindowA', 'DeliveryWindowB']]
timing['DeliveryWindowA_hour'] = pd.to_datetime(timing['DeliveryWindowA']).dt.hour
timing['DeliveryWindowA_minutes'] = pd.to_datetime(timing['DeliveryWindowA']).dt.minute
timing['Earliest'] = 60*timing['DeliveryWindowA_hour'] + timing['DeliveryWindowA_minutes']
timing['DeliveryWindowB_hour'] = pd.to_datetime(timing['DeliveryWindowB']).dt.hour
timing['DeliveryWindowB_minutes'] = pd.to_datetime(timing['DeliveryWindowB']).dt.minute
timing['Latest'] = 60*timing['DeliveryWindowB_hour'] + timing['DeliveryWindowB_minutes']
timing = timing[['Earliest', 'Latest']]

# If time window A == time window B, this means the package can arrive any time before
timing['Earliest'][timing['Earliest'] == timing['Latest']] = 0

timing.loc[-1] = [0, 0]  # adding a row
timing.index = timing.index + 1  # shifting index
timing = timing.sort_index()

timing.loc[np.max(timing.index.values)+1] = [0, 1440]

pd.DataFrame(demand).to_csv("test_demand.csv")
pd.DataFrame(time_matrix).to_csv("test_time.csv")
pd.DataFrame(cost_matrix).to_csv("test_cost.csv")
timing.to_csv("test_timing.csv")

