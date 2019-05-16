import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import osmnx as ox

# os.chdir("/Users/gabrielc/Dropbox (MIT)/15071c_Proj/logistics/data")

df_full = pd.read_csv("courier_trips_with_OD_nodes.csv")
df_full = df_full.drop('Unnamed: 0', axis = 1)
df_full.shape

df_full.head(2)

df = df_full[(df_full['pickup_locality'] == "manhattan") & (df_full['delivery_locality'] == "manhattan")]

df['OrderTrackingID'] = pd.factorize(df['OrderTrackingID'])[0]
df['CustomerID'] = pd.factorize(df['CustomerID'])[0]
df['CourierID'] = pd.factorize(df['CourierID'])[0]


print("UNIQUE VALUES")
for column in df:
    print(column, ": ", df[column].nunique())

print(df.delivery_locality.unique())
print(df.ServiceType.unique())
print(df.VehicleType.unique())

# Why are there more rows than tracking IDs?
count = df.groupby('OrderTrackingID')[['CourierID']].count().sort_values('CourierID', ascending = False)
duplicates = list(count[count['CourierID'] >= 2].index)
duplicates

df[df['OrderTrackingID'] == duplicates[0]]
df[df['OrderTrackingID'] == duplicates[1]]
df[df['OrderTrackingID'] == duplicates[2]]

df = df.drop_duplicates(subset='OrderTrackingID', keep = 'last')

df.shape

ymax = max(df.pickup_lat.append(df.delivery_lat))
ymin = min(df.pickup_lat.append(df.delivery_lat))
xmax = max(df.pickup_lng.append(df.delivery_lng))
xmin = min(df.pickup_lng.append(df.delivery_lng))


g = ox.graph_from_bbox(
    ymax, ymin, xmax, xmin,
    network_type = 'drive'
)

def get_lat(x):
    if g.has_node(x):
        output = g.node[x]['y']
    else:
        output = 0
    return(output)

def get_long(x):
    if g.has_node(x):
        output = g.node[x]['x']
    else:
        output = 0
    return(output)

df['pickup_y'] = df['closest_node_pickup'].apply(get_lat)
df['pickup_x'] = df['closest_node_pickup'].apply(get_long)
df['dropoff_y'] = df['closest_node_delivery'].apply(get_lat)
df['dropoff_x'] = df['closest_node_delivery'].apply(get_long)

df = df[df['pickup_y'] != 0]
df = df[df['pickup_x'] != 0]
df = df[df['dropoff_y'] != 0]
df = df[df['dropoff_x'] != 0]

from matplotlib.pyplot import figure
figure(num=None, figsize=(5, 10), dpi=100, facecolor='w', edgecolor='k')
fig, ax = ox.plot_graph(g, fig_height=80, show=False, close=False, edge_alpha = 0.5, node_alpha = 0)
plt.scatter(x = df['pickup_x'], y = df['pickup_y'], marker = '.', s = 100)

freq = df.groupby('closest_node_pickup')[['OrderTrackingID']].count().sort_values('OrderTrackingID', ascending = False).reset_index()

freq['pickup_y'] = freq['closest_node_pickup'].apply(get_lat)
freq['pickup_x'] = freq['closest_node_pickup'].apply(get_long)

freq['OrderTrackingID'] = 100000*(freq['OrderTrackingID']/np.sum(freq['OrderTrackingID']))

freq.head()

figure(num=None, figsize=(5, 10), dpi=100, facecolor='w', edgecolor='k')
fig, ax = ox.plot_graph(g, fig_height=60, show=False, close=False, edge_alpha = 0.5, node_alpha = 0)
ax.scatter(x=freq['pickup_x'], y=freq['pickup_y'], s = freq['OrderTrackingID'])

df.to_csv("data.csv")

# ### Attempt at Clustering

points = np.array(df[['pickup_y', 'pickup_x']])
rads = np.radians(points)
import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=20, metric='haversine')
cluster_labels = clusterer.fit_predict(points)
df_cluster = df
df_cluster['Cluster'] = cluster_labels
df_cluster.head()
figure(num=None, figsize=(5, 10), dpi=100, facecolor='w', edgecolor='k')
fig, ax = ox.plot_graph(g, fig_height=80, show=False, close=False, edge_alpha = 0.5, node_alpha = 0)
plt.scatter(y=df_cluster['pickup_y'], x = df_cluster['pickup_x'], c = df_cluster['Cluster'], s = 100)

