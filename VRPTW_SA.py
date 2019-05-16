

from __future__ import print_function
import os, glob
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


os.chdir("/Users/gabriel/Dropbox (MIT)/MIT/Spring_2019/1263/SCM_course_project/")


# In[4]:


df = pd.read_csv("data.csv")
df['Date'] = (pd.to_datetime(df['DeliveryWindowA']).dt.date).astype(str)


# In[5]:


date_list = df['Date'].unique()


# In[6]:


results1 = dict()


# In[7]:


date_list.sort()


# In[41]:



# In[42]:


os.chdir("/Users/gabriel/Dropbox (MIT)/MIT/Spring_2019/1263/SCM_course_project/1_CVS/")
for date in date_list:
    print("*********************")
    print(date)
    time_window = pd.read_csv("time_window_" + str(date) + "_.csv")
    # Convert Timewindows from Pandas DataFrame to NumpyInt64 Array to Python Native Array Int
    time_windows = [tuple(x) for x in time_window.values]
    time_windows = [(time[0].item(), time[1].item()) for time in time_windows]
    
    time_matrix = pd.read_csv("time_matrix_" + str(date) + "_.csv")
    N = time_matrix.shape[0]
    # Convert
    time_matrix = np.array(time_matrix).tolist()
    
    demand = np.ones(N).astype(int).tolist()
    demand[0] = 0
    demand[-1] = 0
    
    def create_data_model():
        """Stores the datafor the problem."""
        data = {}
        data['time_matrix'] = time_matrix
        data['time_windows'] = time_windows
        data['num_vehicles'] = 20
        data['vehicle_capacities'] = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
        data['demands'] = demand
        data['depot'] = 0
        return data
    
    def print_solution(data, manager, routing, assignment):
        """Prints assignment on console."""
        time_dimension = routing.GetDimensionOrDie('Time')
        total_time = 0
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
            while not routing.IsEnd(index):
                time_var = time_dimension.CumulVar(index)
                plan_output += '{0} Time({1},{2}) -> '.format(
                    manager.IndexToNode(index), assignment.Min(time_var),
                    assignment.Max(time_var))
                index = assignment.Value(routing.NextVar(index))
            time_var = time_dimension.CumulVar(index)
            plan_output += '{0} Time({1},{2})\n'.format(
                manager.IndexToNode(index), assignment.Min(time_var),
                assignment.Max(time_var))
            plan_output += 'Time of the route: {}min\n'.format(
                assignment.Min(time_var))
            # print(plan_output)
            total_time += assignment.Min(time_var)
        print('Total time of all routes: {}min'.format(total_time))
        return(total_time)
    
    def main():
        """Solve the VRP with time windows."""
        data = create_data_model()
        manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']), data['num_vehicles'], data['depot'])
        routing = pywrapcp.RoutingModel(manager)
    
        def time_callback(from_index, to_index):
            """Returns the travel time between the two nodes."""
            # Convert from routing variable Index to time matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['time_matrix'][from_node][to_node]
        
       def demand_callback(from_index):
           """Returns the demand of the node."""
           # Convert from routing variable Index to demands NodeIndex.
           from_node = manager.IndexToNode(from_index)
           return data['demands'][from_node]
    
        transit_callback_index = routing.RegisterTransitCallback(time_callback)
        demand_callback_index = routing.RegisterUnaryTransitCallback(
             demand_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        time = 'Time'
        routing.AddDimension(
            transit_callback_index,
            60,  # allow waiting time
            1440,  # maximum time per vehicle
            False,  # Don't force start cumul to zero.
            time)
       routing.AddDimensionWithVehicleCapacity(
           demand_callback_index,
           0,  # null capacity slack
           data['vehicle_capacities'],  # vehicle maximum capacities
           True,  # start cumul to zero
           'Capacity')
        time_dimension = routing.GetDimensionOrDie(time)
        # Add time window constraints for each location except depot.
        for location_idx, time_window in enumerate(data['time_windows']):
            if location_idx == 0:
                continue
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
        # Add time window constraints for each vehicle start node.
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            time_dimension.CumulVar(index).SetRange(data['time_windows'][0][0], data['time_windows'][0][1])
        for i in range(data['num_vehicles']):
            routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.Start(i)))
            routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.End(i)))
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING)
        search_parameters.time_limit.seconds = 100
        search_parameters.log_search = True
        assignment = routing.SolveWithParameters(search_parameters)
        

        if assignment:
            time_wait = print_solution(data, manager, routing, assignment)
            results1[date] = time_wait
            print(results1[date])
    
    main()


# In[43]:


pd.DataFrame(results1, index=['cost']).transpose().to_csv("CVS.csv")


# In[44]:


results2 = dict()


# In[45]:


os.chdir("/Users/gabriel/Dropbox (MIT)/MIT/Spring_2019/1263/SCM_course_project/3_Walgreens/")
for date in date_list:
    print("*********************")
    print(date)
    time_window = pd.read_csv("time_window_" + str(date) + "_.csv")
    # Convert Timewindows from Pandas DataFrame to NumpyInt64 Array to Python Native Array Int
    time_windows = [tuple(x) for x in time_window.values]
    time_windows = [(time[0].item(), time[1].item()) for time in time_windows]
    
    time_matrix = pd.read_csv("time_matrix_" + str(date) + "_.csv")
    N = time_matrix.shape[0]
    # Convert
    time_matrix = np.array(time_matrix).tolist()
    
    demand = np.ones(N).astype(int).tolist()
    demand[0] = 0
    demand[-1] = 0
    
    def create_data_model():
        """Stores the datafor the problem."""
        data = {}
        data['time_matrix'] = time_matrix
        data['time_windows'] = time_windows
        data['num_vehicles'] = 20
        # data['vehicle_capacities'] = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
        # data['demands'] = demand
        data['depot'] = 0
        return data
    
    def print_solution(data, manager, routing, assignment):
        """Prints assignment on console."""
        time_dimension = routing.GetDimensionOrDie('Time')
        total_time = 0
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
            while not routing.IsEnd(index):
                time_var = time_dimension.CumulVar(index)
                plan_output += '{0} Time({1},{2}) -> '.format(
                    manager.IndexToNode(index), assignment.Min(time_var),
                    assignment.Max(time_var))
                index = assignment.Value(routing.NextVar(index))
            time_var = time_dimension.CumulVar(index)
            plan_output += '{0} Time({1},{2})\n'.format(
                manager.IndexToNode(index), assignment.Min(time_var),
                assignment.Max(time_var))
            plan_output += 'Time of the route: {}min\n'.format(
                assignment.Min(time_var))
            # print(plan_output)
            total_time += assignment.Min(time_var)
        print('Total time of all routes: {}min'.format(total_time))
        return(total_time)
    
    def main():
        """Solve the VRP with time windows."""
        data = create_data_model()
        manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']), data['num_vehicles'], data['depot'])
        routing = pywrapcp.RoutingModel(manager)
    
        def time_callback(from_index, to_index):
            """Returns the travel time between the two nodes."""
            # Convert from routing variable Index to time matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['time_matrix'][from_node][to_node]
        
       def demand_callback(from_index):
           """Returns the demand of the node."""
           # Convert from routing variable Index to demands NodeIndex.
           from_node = manager.IndexToNode(from_index)
           return data['demands'][from_node]
    
        transit_callback_index = routing.RegisterTransitCallback(time_callback)
        demand_callback_index = routing.RegisterUnaryTransitCallback(
            demand_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        time = 'Time'
        routing.AddDimension(
            transit_callback_index,
            60,  # allow waiting time
            1440,  # maximum time per vehicle
            False,  # Don't force start cumul to zero.
            time)
      routing.AddDimensionWithVehicleCapacity(
          demand_callback_index,
          0,  # null capacity slack
          data['vehicle_capacities'],  # vehicle maximum capacities
          True,  # start cumul to zero
          'Capacity')
        time_dimension = routing.GetDimensionOrDie(time)
        # Add time window constraints for each location except depot.
        for location_idx, time_window in enumerate(data['time_windows']):
            if location_idx == 0:
                continue
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
        # Add time window constraints for each vehicle start node.
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            time_dimension.CumulVar(index).SetRange(data['time_windows'][0][0], data['time_windows'][0][1])
        for i in range(data['num_vehicles']):
            routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.Start(i)))
            routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.End(i)))
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING)
        search_parameters.time_limit.seconds = 100
        search_parameters.log_search = True
        assignment = routing.SolveWithParameters(search_parameters)
        

        if assignment:
            time_wait = print_solution(data, manager, routing, assignment)
            results2[date] = time_wait
            print(results2[date])
    
    main()


# In[47]:


pd.DataFrame(results2, index=['cost']).transpose().to_csv("walgreens.csv")


# In[48]:


pd.DataFrame(results2, index=['cost']).transpose()


# In[49]:


results3 = dict()


# In[51]:


os.chdir("/Users/gabriel/Dropbox (MIT)/MIT/Spring_2019/1263/SCM_course_project/0_Status_Quo/")
for date in date_list:
    print("******************************************")
    print(date)
    time_window = pd.read_csv("time_window_" + str(date) + "_.csv")
    # Convert Timewindows from Pandas DataFrame to NumpyInt64 Array to Python Native Array Int
    time_windows = [tuple(x) for x in time_window.values]
    time_windows = [(time[0].item(), time[1].item()) for time in time_windows]
    
    time_matrix = pd.read_csv("time_matrix_" + str(date) + "_.csv")
    N = time_matrix.shape[0]
    # Convert
    time_matrix = np.array(time_matrix).tolist()
    
    demand = np.ones(N).astype(int).tolist()
    demand[0] = 0
    demand[-1] = 0
    
    def create_data_model():
        """Stores the datafor the problem."""
        data = {}
        data['time_matrix'] = time_matrix
        data['time_windows'] = time_windows
        data['num_vehicles'] = 20
        data['vehicle_capacities'] = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
        data['demands'] = demand
        data['depot'] = 0
        return data
    
    def print_solution(data, manager, routing, assignment):
        """Prints assignment on console."""
        time_dimension = routing.GetDimensionOrDie('Time')
        total_time = 0
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
            while not routing.IsEnd(index):
                time_var = time_dimension.CumulVar(index)
                plan_output += '{0} Time({1},{2}) -> '.format(
                    manager.IndexToNode(index), assignment.Min(time_var),
                    assignment.Max(time_var))
                index = assignment.Value(routing.NextVar(index))
            time_var = time_dimension.CumulVar(index)
            plan_output += '{0} Time({1},{2})\n'.format(
                manager.IndexToNode(index), assignment.Min(time_var),
                assignment.Max(time_var))
            plan_output += 'Time of the route: {}min\n'.format(
                assignment.Min(time_var))
            # print(plan_output)
            total_time += assignment.Min(time_var)
        print('Total time of all routes: {}min'.format(total_time))
        return(total_time)
    
    def main():
        """Solve the VRP with time windows."""
        data = create_data_model()
        manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']), data['num_vehicles'], data['depot'])
        routing = pywrapcp.RoutingModel(manager)
    
        def time_callback(from_index, to_index):
            """Returns the travel time between the two nodes."""
            # Convert from routing variable Index to time matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['time_matrix'][from_node][to_node]
        
       def demand_callback(from_index):
           """Returns the demand of the node."""
           # Convert from routing variable Index to demands NodeIndex.
           from_node = manager.IndexToNode(from_index)
           return data['demands'][from_node]
    
        transit_callback_index = routing.RegisterTransitCallback(time_callback)
        demand_callback_index = routing.RegisterUnaryTransitCallback(
            demand_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        time = 'Time'
        routing.AddDimension(
            transit_callback_index,
            60,  # allow waiting time
            1440,  # maximum time per vehicle
            False,  # Don't force start cumul to zero.
            time)
       # routing.AddDimensionWithVehicleCapacity(
       #     demand_callback_index,
       #     0,  # null capacity slack
       #     data['vehicle_capacities'],  # vehicle maximum capacities
       #     True,  # start cumul to zero
       #     'Capacity')
        time_dimension = routing.GetDimensionOrDie(time)
        # Add time window constraints for each location except depot.
        for location_idx, time_window in enumerate(data['time_windows']):
            if location_idx == 0:
                continue
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
        # Add time window constraints for each vehicle start node.
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            time_dimension.CumulVar(index).SetRange(data['time_windows'][0][0], data['time_windows'][0][1])
        for i in range(data['num_vehicles']):
            routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.Start(i)))
            routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.End(i)))
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING)
        search_parameters.time_limit.seconds = 100
        search_parameters.log_search = True
        assignment = routing.SolveWithParameters(search_parameters)
        

        if assignment:
            time_wait = print_solution(data, manager, routing, assignment)
            results3[date] = time_wait
            print(results3[date])
    
    main()


# In[55]:


pd.DataFrame(results3, index=['cost']).transpose().to_csv("SQ.csv")


# In[52]:


results4 = dict()


# In[ ]:


[163:]:


# In[71]:


date_list[352]


# In[68]:


results4


# In[72]:


os.chdir("/Users/gabriel/Dropbox (MIT)/MIT/Spring_2019/1263/SCM_course_project/1_CVS/")
for date in date_list[352:]:
    print("*********************")
    print(date)
    time_window = pd.read_csv("time_window_" + str(date) + "_.csv")
    # Convert Timewindows from Pandas DataFrame to NumpyInt64 Array to Python Native Array Int
    time_windows = [tuple(x) for x in time_window.values]
    time_windows = [(time[0].item(), time[1].item()) for time in time_windows]
    
    time_matrix = pd.read_csv("time_matrix_" + str(date) + "_.csv")
    N = time_matrix.shape[0]
    # Convert
    time_matrix = np.array(time_matrix).tolist()
    
    demand = np.ones(N).astype(int).tolist()
    demand[0] = 0
    demand[-1] = 0
    
    def create_data_model():
        """Stores the datafor the problem."""
        data = {}
        data['time_matrix'] = time_matrix
        data['time_windows'] = time_windows
        data['num_vehicles'] = 20
        # data['vehicle_capacities'] = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
        # data['demands'] = demand
        data['depot'] = 0
        return data
    
    def print_solution(data, manager, routing, assignment):
        """Prints assignment on console."""
        time_dimension = routing.GetDimensionOrDie('Time')
        total_time = 0
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
            while not routing.IsEnd(index):
                time_var = time_dimension.CumulVar(index)
                plan_output += '{0} Time({1},{2}) -> '.format(
                    manager.IndexToNode(index), assignment.Min(time_var),
                    assignment.Max(time_var))
                index = assignment.Value(routing.NextVar(index))
            time_var = time_dimension.CumulVar(index)
            plan_output += '{0} Time({1},{2})\n'.format(
                manager.IndexToNode(index), assignment.Min(time_var),
                assignment.Max(time_var))
            plan_output += 'Time of the route: {}min\n'.format(
                assignment.Min(time_var))
            # print(plan_output)
            total_time += assignment.Min(time_var)
        print('Total time of all routes: {}min'.format(total_time))
        return(total_time)
    
    def main():
        """Solve the VRP with time windows."""
        data = create_data_model()
        manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']), data['num_vehicles'], data['depot'])
        routing = pywrapcp.RoutingModel(manager)
    
        def time_callback(from_index, to_index):
            """Returns the travel time between the two nodes."""
            # Convert from routing variable Index to time matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['time_matrix'][from_node][to_node]
        
       # def demand_callback(from_index):
       #     """Returns the demand of the node."""
       #     # Convert from routing variable Index to demands NodeIndex.
       #     from_node = manager.IndexToNode(from_index)
       #     return data['demands'][from_node]
    
        transit_callback_index = routing.RegisterTransitCallback(time_callback)
       #  demand_callback_index = routing.RegisterUnaryTransitCallback(
       #      demand_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        time = 'Time'
        routing.AddDimension(
            transit_callback_index,
            60,  # allow waiting time
            1440,  # maximum time per vehicle
            False,  # Don't force start cumul to zero.
            time)
       # routing.AddDimensionWithVehicleCapacity(
       #     demand_callback_index,
       #     0,  # null capacity slack
       #     data['vehicle_capacities'],  # vehicle maximum capacities
       #     True,  # start cumul to zero
       #     'Capacity')
        time_dimension = routing.GetDimensionOrDie(time)
        # Add time window constraints for each location except depot.
        for location_idx, time_window in enumerate(data['time_windows']):
            if location_idx == 0:
                continue
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
        # Add time window constraints for each vehicle start node.
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            time_dimension.CumulVar(index).SetRange(data['time_windows'][0][0], data['time_windows'][0][1])
        for i in range(data['num_vehicles']):
            routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.Start(i)))
            routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.End(i)))
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING)
        search_parameters.time_limit.seconds = 100
        search_parameters.log_search = True
        assignment = routing.SolveWithParameters(search_parameters)
        

        if assignment:
            time_wait = print_solution(data, manager, routing, assignment)
            results4[date] = time_wait
            print(results4[date])
    
    main()


# In[73]:


pd.DataFrame(results4, index=['cost']).transpose().to_csv("CVS_full.csv")


# In[77]:


results4_df = pd.DataFrame(results4, index=['cost']).transpose()


# In[82]:


new_date_list = list(results4_df.index)


# In[83]:


results5 = dict()


# In[84]:


os.chdir("/Users/gabriel/Dropbox (MIT)/MIT/Spring_2019/1263/SCM_course_project/3_Walgreens/")
for date in new_date_list:
    print("*********************")
    print(date)
    time_window = pd.read_csv("time_window_" + str(date) + "_.csv")
    # Convert Timewindows from Pandas DataFrame to NumpyInt64 Array to Python Native Array Int
    time_windows = [tuple(x) for x in time_window.values]
    time_windows = [(time[0].item(), time[1].item()) for time in time_windows]
    
    time_matrix = pd.read_csv("time_matrix_" + str(date) + "_.csv")
    N = time_matrix.shape[0]
    # Convert
    time_matrix = np.array(time_matrix).tolist()
    
    demand = np.ones(N).astype(int).tolist()
    demand[0] = 0
    demand[-1] = 0
    
    def create_data_model():
        """Stores the datafor the problem."""
        data = {}
        data['time_matrix'] = time_matrix
        data['time_windows'] = time_windows
        data['num_vehicles'] = 20
        # data['vehicle_capacities'] = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
        # data['demands'] = demand
        data['depot'] = 0
        return data
    
    def print_solution(data, manager, routing, assignment):
        """Prints assignment on console."""
        time_dimension = routing.GetDimensionOrDie('Time')
        total_time = 0
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
            while not routing.IsEnd(index):
                time_var = time_dimension.CumulVar(index)
                plan_output += '{0} Time({1},{2}) -> '.format(
                    manager.IndexToNode(index), assignment.Min(time_var),
                    assignment.Max(time_var))
                index = assignment.Value(routing.NextVar(index))
            time_var = time_dimension.CumulVar(index)
            plan_output += '{0} Time({1},{2})\n'.format(
                manager.IndexToNode(index), assignment.Min(time_var),
                assignment.Max(time_var))
            plan_output += 'Time of the route: {}min\n'.format(
                assignment.Min(time_var))
            # print(plan_output)
            total_time += assignment.Min(time_var)
        print('Total time of all routes: {}min'.format(total_time))
        return(total_time)
    
    def main():
        """Solve the VRP with time windows."""
        data = create_data_model()
        manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']), data['num_vehicles'], data['depot'])
        routing = pywrapcp.RoutingModel(manager)
    
        def time_callback(from_index, to_index):
            """Returns the travel time between the two nodes."""
            # Convert from routing variable Index to time matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['time_matrix'][from_node][to_node]
        
       # def demand_callback(from_index):
       #     """Returns the demand of the node."""
       #     # Convert from routing variable Index to demands NodeIndex.
       #     from_node = manager.IndexToNode(from_index)
       #     return data['demands'][from_node]
    
        transit_callback_index = routing.RegisterTransitCallback(time_callback)
       #  demand_callback_index = routing.RegisterUnaryTransitCallback(
       #      demand_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        time = 'Time'
        routing.AddDimension(
            transit_callback_index,
            60,  # allow waiting time
            1440,  # maximum time per vehicle
            False,  # Don't force start cumul to zero.
            time)
       # routing.AddDimensionWithVehicleCapacity(
       #     demand_callback_index,
       #     0,  # null capacity slack
       #     data['vehicle_capacities'],  # vehicle maximum capacities
       #     True,  # start cumul to zero
       #     'Capacity')
        time_dimension = routing.GetDimensionOrDie(time)
        # Add time window constraints for each location except depot.
        for location_idx, time_window in enumerate(data['time_windows']):
            if location_idx == 0:
                continue
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
        # Add time window constraints for each vehicle start node.
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            time_dimension.CumulVar(index).SetRange(data['time_windows'][0][0], data['time_windows'][0][1])
        for i in range(data['num_vehicles']):
            routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.Start(i)))
            routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.End(i)))
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING)
        search_parameters.time_limit.seconds = 100
        search_parameters.log_search = True
        assignment = routing.SolveWithParameters(search_parameters)
        

        if assignment:
            time_wait = print_solution(data, manager, routing, assignment)
            results5[date] = time_wait
            print(results5[date])
    
    main()


# In[85]:


pd.DataFrame(results5, index=['cost']).transpose().to_csv("walgreens_full.csv")


# In[86]:


results6 = dict()


# In[110]:


new_date_list[264]


# In[111]:


os.chdir("/Users/gabriel/Dropbox (MIT)/MIT/Spring_2019/1263/SCM_course_project/0_Status_Quo/")
for date in new_date_list[266:]:
    print("******************************************")
    print(date)
    time_window = pd.read_csv("time_window_" + str(date) + "_.csv")
    # Convert Timewindows from Pandas DataFrame to NumpyInt64 Array to Python Native Array Int
    time_windows = [tuple(x) for x in time_window.values]
    time_windows = [(time[0].item(), time[1].item()) for time in time_windows]
    
    time_matrix = pd.read_csv("time_matrix_" + str(date) + "_.csv")
    N = time_matrix.shape[0]
    # Convert
    time_matrix = np.array(time_matrix).tolist()
    
    demand = np.ones(N).astype(int).tolist()
    demand[0] = 0
    demand[-1] = 0
    
    def create_data_model():
        """Stores the datafor the problem."""
        data = {}
        data['time_matrix'] = time_matrix
        data['time_windows'] = time_windows
        data['num_vehicles'] = 20
        # data['vehicle_capacities'] = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
        # data['demands'] = demand
        data['depot'] = 0
        return data
    
    def print_solution(data, manager, routing, assignment):
        """Prints assignment on console."""
        time_dimension = routing.GetDimensionOrDie('Time')
        total_time = 0
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
            while not routing.IsEnd(index):
                time_var = time_dimension.CumulVar(index)
                plan_output += '{0} Time({1},{2}) -> '.format(
                    manager.IndexToNode(index), assignment.Min(time_var),
                    assignment.Max(time_var))
                index = assignment.Value(routing.NextVar(index))
            time_var = time_dimension.CumulVar(index)
            plan_output += '{0} Time({1},{2})\n'.format(
                manager.IndexToNode(index), assignment.Min(time_var),
                assignment.Max(time_var))
            plan_output += 'Time of the route: {}min\n'.format(
                assignment.Min(time_var))
            # print(plan_output)
            total_time += assignment.Min(time_var)
        print('Total time of all routes: {}min'.format(total_time))
        return(total_time)
    
    def main():
        """Solve the VRP with time windows."""
        data = create_data_model()
        manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']), data['num_vehicles'], data['depot'])
        routing = pywrapcp.RoutingModel(manager)
    
        def time_callback(from_index, to_index):
            """Returns the travel time between the two nodes."""
            # Convert from routing variable Index to time matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['time_matrix'][from_node][to_node]
        
       # def demand_callback(from_index):
       #     """Returns the demand of the node."""
       #     # Convert from routing variable Index to demands NodeIndex.
       #     from_node = manager.IndexToNode(from_index)
       #     return data['demands'][from_node]
    
        transit_callback_index = routing.RegisterTransitCallback(time_callback)
       #  demand_callback_index = routing.RegisterUnaryTransitCallback(
       #      demand_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        time = 'Time'
        routing.AddDimension(
            transit_callback_index,
            60,  # allow waiting time
            1440,  # maximum time per vehicle
            False,  # Don't force start cumul to zero.
            time)
       # routing.AddDimensionWithVehicleCapacity(
       #     demand_callback_index,
       #     0,  # null capacity slack
       #     data['vehicle_capacities'],  # vehicle maximum capacities
       #     True,  # start cumul to zero
       #     'Capacity')
        time_dimension = routing.GetDimensionOrDie(time)
        # Add time window constraints for each location except depot.
        for location_idx, time_window in enumerate(data['time_windows']):
            if location_idx == 0:
                continue
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
        # Add time window constraints for each vehicle start node.
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            time_dimension.CumulVar(index).SetRange(data['time_windows'][0][0], data['time_windows'][0][1])
        for i in range(data['num_vehicles']):
            routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.Start(i)))
            routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.End(i)))
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING)
        search_parameters.time_limit.seconds = 100
        search_parameters.log_search = True
        assignment = routing.SolveWithParameters(search_parameters)
        

        if assignment:
            time_wait = print_solution(data, manager, routing, assignment)
            results6[date] = time_wait
            print(results6[date])
    
    main()


# In[112]:


pd.DataFrame(results6, index=['cost']).transpose().to_csv("SQ_full.csv")


# In[101]:


results6


# In[ ]:





# In[ ]:




