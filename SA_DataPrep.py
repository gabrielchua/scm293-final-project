import os
import pandas as pd
import numpy as np
from haversine import haversine

os.chdir("/Users/gabrielc/Dropbox (MIT)/MIT/Spring_2019/1263/SCM_course_project/")

# Import Data
df = pd.read_csv("data.csv")
df['DropOffLatLon'] = list(zip(df['dropoff_y'], df['dropoff_x']))

walgreens = pd.read_csv("store_coordinates/cvs.csv")

walgreens['LatLon'] = list(zip(walgreens['Latitude'], walgreens['Longitude']))

def nearest_store(coord, list_of_stores):
	temp_list = []
	list_of_stores = np.array(list_of_stores)
	for i in list_of_stores:
		temp_list.append(haversine(coord, i))
	temp_list = np.array(temp_list)
	coord = list_of_stores[np.argmin(temp_list)]
	return coord

Dropoff_points = df['DropOffLatLon'].unique()
Dropoff_points = pd.DataFrame(Dropoff_points)
Dropoff_points.columns = ['DropOffLatLon']
Dropoff_points['Nearest_Store_Lat'] = 1.0
Dropoff_points['Nearest_Store_Lon'] = 1.0

for i in range(0, Dropoff_points.shape[0]):
	if i % 200 == 0:
		print(i)
	temp_coord = nearest_store(Dropoff_points['DropOffLatLon'][i], walgreens['LatLon'])
	Dropoff_points['Nearest_Store_Lat'][i] = temp_coord[0]
	Dropoff_points['Nearest_Store_Lon'][i] = temp_coord[1]

Dropoff_points['Nearest_LatLon'] = list(zip(Dropoff_points['Nearest_Store_Lat'],Dropoff_points['Nearest_Store_Lon']))
Dropoff_points = Dropoff_points[['DropOffLatLon','Nearest_LatLon', 'Nearest_Store_Lat', 'Nearest_Store_Lon']]
Dropoff_points['Distance'] = 0.0

for i in range(0, Dropoff_points.shape[0]):
	Dropoff_points['Distance'][i] = haversine(Dropoff_points['DropOffLatLon'][i], Dropoff_points['Nearest_LatLon'][i], unit = 'm')


df_merged = df.merge(Dropoff_points, on = 'DropOffLatLon')
df_merged['Distance'] = df_merged['Distance']*0.001
df_merged['Prob'] = np.exp(-df_merged['Distance'])
np.min(df_merged['Prob'])
np.median(df_merged['Prob'])
np.mean(df_merged['Prob'])
np.max(df_merged['Prob'])

df_merged['Go_Nearest'] = np.random.binomial(1, df_merged['Prob'], size = len(df_merged['Prob']))
sum(df_merged['Go_Nearest'])/len(df_merged['Go_Nearest'])
# 67% would use it

df_merged['dropoff_y'][df_merged['Go_Nearest'] == 1] = df_merged['Nearest_Store_Lat'][df_merged['Go_Nearest'] == 1]
df_merged['dropoff_x'][df_merged['Go_Nearest'] == 1] = df_merged['Nearest_Store_Lon'][df_merged['Go_Nearest'] == 1]

df_merged.to_csv("cvs_option.csv")





walgreens = pd.read_csv("store_coordinates/walgreens.csv")

walgreens['FullAddress'] = walgreens['Store'] + [" "] + walgreens['ZipCode'].map(str)
from geopy.geocoders import Nominatim

walgreens['Lat'] = 1.0
walgreens['Lon'] = 1.0

for i in range(0,len(walgreens['FullAddress'])):
	print(i)
	geolocator = Nominatim()
	location = geolocator.geocode(walgreens['FullAddress'][i])
	walgreens.iloc[i,3] = location.latitude
	walgreens.iloc[i,4] = location.longitude

walgreens['LatLon'] = list(zip(walgreens['Lat'], walgreens['Lon']))

df = pd.read_csv("data.csv")
df['DropOffLatLon'] = list(zip(df['dropoff_y'], df['dropoff_x']))
Dropoff_points = df['DropOffLatLon'].unique()
Dropoff_points = pd.DataFrame(Dropoff_points)
Dropoff_points.columns = ['DropOffLatLon']
Dropoff_points['Nearest_Store_Lat'] = 1.0
Dropoff_points['Nearest_Store_Lon'] = 1.0

for i in range(0, Dropoff_points.shape[0]):
	if i % 200 == 0:
		print(i)
	temp_coord = nearest_store(Dropoff_points['DropOffLatLon'][i], walgreens['LatLon'])
	Dropoff_points['Nearest_Store_Lat'][i] = temp_coord[0]
	Dropoff_points['Nearest_Store_Lon'][i] = temp_coord[1]

Dropoff_points['Nearest_LatLon'] = list(zip(Dropoff_points['Nearest_Store_Lat'],Dropoff_points['Nearest_Store_Lon']))
Dropoff_points = Dropoff_points[['DropOffLatLon','Nearest_LatLon', 'Nearest_Store_Lat', 'Nearest_Store_Lon']]
Dropoff_points['Distance'] = 0.0

for i in range(0, Dropoff_points.shape[0]):
	Dropoff_points['Distance'][i] = haversine(Dropoff_points['DropOffLatLon'][i], Dropoff_points['Nearest_LatLon'][i], unit = 'm')

df_merged = df.merge(Dropoff_points, on = 'DropOffLatLon')
df_merged['Distance'] = df_merged['Distance']*0.0005
df_merged['Prob'] = np.exp(-df_merged['Distance'])
np.min(df_merged['Prob'])
np.median(df_merged['Prob'])
np.mean(df_merged['Prob'])
np.max(df_merged['Prob'])

df_merged['Go_Nearest'] = np.random.binomial(1, df_merged['Prob'], size = len(df_merged['Prob']))
sum(df_merged['Go_Nearest'])/len(df_merged['Go_Nearest'])
# 83% would use it

df_merged['dropoff_y'][df_merged['Go_Nearest'] == 1] = df_merged['Nearest_Store_Lat'][df_merged['Go_Nearest'] == 1]
df_merged['dropoff_x'][df_merged['Go_Nearest'] == 1] = df_merged['Nearest_Store_Lon'][df_merged['Go_Nearest'] == 1]

df_merged.to_csv("walgreen_option.csv")


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

Dates = df['Date'].unique()

for k in Dates:
	print(k)
	df2 = df[df['Date'] == k].reset_index()
	df2['CC'] = 0
	df2['CC'][(df2['dropoff_y'] == df2['Nearest_Store_Lat']) & (df2['dropoff_x'] == df2['Nearest_Store_Lon'])] = 1
	df2 = df2[['DeliveryWindowA', 'DeliveryWindowB', 'dropoff_y', 'dropoff_x', 'CC']]
	
	# Dropoff Nodes & Add Nodes
	df2['dropoff'] = list(zip(df2['dropoff_y'], df2['dropoff_x']))
	
	dropoff_points = list(zip(df2['dropoff_y'], df2['dropoff_x']))
	dropoff_points = [(40.758994, -73.999744)] + dropoff_points + [(40.758993, -73.999743)]
	
	# Factorize the nodes 
	# Eventually save the node_dict too!
	nodes, node_dict = pd.factorize(dropoff_points)
	number_of_nodes = len(nodes)
	
	# demand = np.ones(number_of_nodes-2)
	# demand = np.insert(demand, 0, 0)
	# demand = np.append(demand, 0)
	
	distance_matrix = np.zeros([number_of_nodes, number_of_nodes])
	
	for i in range(0, number_of_nodes):
	    for j in range(0, number_of_nodes):
	        if i == j:
	            distance_matrix[i,j] = 0
	        else:
	            distance_matrix[i,j] = haversine(node_dict[nodes[i]], node_dict[nodes[j]], unit='mi')
	
	# cost_matrix = distance_matrix * 5
	# cost_matrix = np.round(cost_matrix)
	
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
	timing['Earliest'][df2['CC'] == 1] = 0
	
	
	timing.loc[-1] = [0, 0]  # adding a row
	timing.index = timing.index + 1  # shifting index
	timing = timing.sort_index()
	
	timing.loc[np.max(timing.index.values)+1] = [0, 1440]
	
	#pd.DataFrame(demand).to_csv("Walgreen_demand.csv")
	pd.DataFrame(time_matrix).to_csv("1_CVS/time_matrix_" + k +"_.csv", index = False)
	# pd.DataFrame(cost_matrix).to_csv("Walgreen_cost.csv")
	timing.to_csv("1_CVS/time_window_" + k +"_.csv", index = False)

import os
import pandas as pd
import numpy as np
from haversine import haversine

os.chdir("/Users/gabrielc/Dropbox (MIT)/MIT/Spring_2019/1263/SCM_course_project")

# Import Data
# df = pd.read_csv("data.csv")
df = pd.read_csv("walgreen_option.csv")

# Filter Date
df['Date'] = (pd.to_datetime(df['DeliveryWindowA']).dt.date).astype(str)

Dates = df['Date'].unique()

for k in Dates:
	print(k)
	df2 = df[df['Date'] == k].reset_index()
	df2['CC'] = 0
	df2['CC'][(df2['dropoff_y'] == df2['Nearest_Store_Lat']) & (df2['dropoff_x'] == df2['Nearest_Store_Lon'])] = 1
	df2 = df2[['DeliveryWindowA', 'DeliveryWindowB', 'dropoff_y', 'dropoff_x', 'CC']]
	
	# Dropoff Nodes & Add Nodes
	df2['dropoff'] = list(zip(df2['dropoff_y'], df2['dropoff_x']))
	
	dropoff_points = list(zip(df2['dropoff_y'], df2['dropoff_x']))
	dropoff_points = [(40.758994, -73.999744)] + dropoff_points + [(40.758993, -73.999743)]
	
	# Factorize the nodes 
	# Eventually save the node_dict too!
	nodes, node_dict = pd.factorize(dropoff_points)
	number_of_nodes = len(nodes)
	
	# demand = np.ones(number_of_nodes-2)
	# demand = np.insert(demand, 0, 0)
	# demand = np.append(demand, 0)
	
	distance_matrix = np.zeros([number_of_nodes, number_of_nodes])
	
	for i in range(0, number_of_nodes):
	    for j in range(0, number_of_nodes):
	        if i == j:
	            distance_matrix[i,j] = 0
	        else:
	            distance_matrix[i,j] = haversine(node_dict[nodes[i]], node_dict[nodes[j]], unit='mi')
	
	# cost_matrix = distance_matrix * 5
	# cost_matrix = np.round(cost_matrix)
	
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
	timing['Earliest'][df2['CC'] == 1] = 0
	timing.loc[-1] = [0, 0]  # adding a row
	timing.index = timing.index + 1  # shifting index
	timing = timing.sort_index()
	
	timing.loc[np.max(timing.index.values)+1] = [0, 1440]
	
	#pd.DataFrame(demand).to_csv("Walgreen_demand.csv")
	pd.DataFrame(time_matrix).to_csv("3_Walgreens/time_matrix_" + k +"_.csv", index = False)
	# pd.DataFrame(cost_matrix).to_csv("Walgreen_cost.csv")
	timing.to_csv("3_Walgreens/time_window_" + k +"_.csv", index = False)

import os
import pandas as pd
import numpy as np
from haversine import haversine

os.chdir("/Users/gabrielc/Dropbox (MIT)/MIT/Spring_2019/1263/SCM_course_project")

# Import Data
# df = pd.read_csv("data.csv")
df = pd.read_csv("data.csv")

# Filter Date
df['Date'] = (pd.to_datetime(df['DeliveryWindowA']).dt.date).astype(str)

Dates = df['Date'].unique()

for k in Dates:
	print(k)
	df2 = df[df['Date'] == k].reset_index()
	df2 = df2[['DeliveryWindowA', 'DeliveryWindowB', 'dropoff_y', 'dropoff_x']]
	
	# Dropoff Nodes & Add Nodes
	df2['dropoff'] = list(zip(df2['dropoff_y'], df2['dropoff_x']))
	
	dropoff_points = list(zip(df2['dropoff_y'], df2['dropoff_x']))
	dropoff_points = [(40.758994, -73.999744)] + dropoff_points + [(40.758993, -73.999743)]
	
	# Factorize the nodes 
	# Eventually save the node_dict too!
	nodes, node_dict = pd.factorize(dropoff_points)
	number_of_nodes = len(nodes)
	
	# demand = np.ones(number_of_nodes-2)
	# demand = np.insert(demand, 0, 0)
	# demand = np.append(demand, 0)
	
	distance_matrix = np.zeros([number_of_nodes, number_of_nodes])
	
	for i in range(0, number_of_nodes):
	    for j in range(0, number_of_nodes):
	        if i == j:
	            distance_matrix[i,j] = 0
	        else:
	            distance_matrix[i,j] = haversine(node_dict[nodes[i]], node_dict[nodes[j]], unit='mi')
	
	# cost_matrix = distance_matrix * 5
	# cost_matrix = np.round(cost_matrix)
	
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
	
	#pd.DataFrame(demand).to_csv("Walgreen_demand.csv")
	pd.DataFrame(time_matrix).to_csv("0_Status_Quo/time_matrix_" + k +"_.csv", index = False)
	# pd.DataFrame(cost_matrix).to_csv("Walgreen_cost.csv")
	timing.to_csv("0_Status_Quo/time_window_" + k +"_.csv", index = False)
