import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import heapq

from google.colab import drive
drive.mount('/content/drive')

def prep(graph_times):
  #Required dataframes are fetched and prepared:
  #pos_data (Node pairs and their positions)
  #graph_times (Table with edge_names and the corresponding times (still example times Cluster 02))
  #factors (Node pairs and the weights with which the times need to be multiplied)"

  pos_data = pd.read_csv("/content/drive/MyDrive/ProO/data/graph_cleaned_weighted.csv")
  pos_data = pos_data.drop_duplicates()

  graph_times = graph_times.drop(columns=graph_times.columns[0])
  g = ["Edge name", "times"]
  graph_times.columns = g

  h = ["Node A", "Node B", "Counter", "Counter_normalized"]
  factors = pd.read_csv("/content/drive/MyDrive/ProO/data/Knotenpaare_normalisiert.csv", header=None, names=h)
  factors = factors.drop(index=0)

  #Merge graph_times and pos_data on edge_names
  pos_data["Edge name"] = pos_data["Edge name"].astype(str)
  graph_times["Edge name"] = graph_times["Edge name"].astype(str)
  merged = pd.merge(graph_times, pos_data, on="Edge name")
  #Create the column "Pairs" in merged to merge with factors (Int-Tupel)
  merged["Node A"] = merged["Node A"].astype(int)
  merged["Node B"] = merged["Node B"].astype(int)
  merged["Pairs"] = list(zip(merged['Node A'], merged['Node B']))
  #Create the column "Pairs" in factors to merge with merged (Int-Tupel)
  factors["Node A"] = factors["Node A"].astype(int)
  factors["Node B"] = factors["Node B"].astype(int)
  factors["Pairs"] = list(zip(factors['Node A'], factors['Node B']))
  factors = factors.drop(columns=['Node A', 'Node B'])

  #Second merge and preparing the columns for the adjacency matrix
  merged_again = pd.merge(merged, factors, on="Pairs")
  merged_again = merged_again.drop(columns=["X of A", "Y of A", "X of B", "Y of B", "Pairs"])
  merged_again["Counter_normalized"] = merged_again["Counter_normalized"].astype(float)
  merged_again["times"] = merged_again["times"].astype(float)
  merged_again["Node A"] = merged_again["Node A"].astype(int)
  merged_again["Node B"] = merged_again["Node B"].astype(int)
  merged_again["weights"] = merged_again["Counter_normalized"] * merged_again["times"]

  return merged_again

def adjacency(tab):
    #Get the matrix dimensions
    n = max(max(list(tab["Node A"])), max(list(tab["Node B"])))

    #Initialize an empty matrix
    adj_matrix = np.zeros((n, n, 2))

    #Fill the matrix with values
    for _, row in tab.iterrows():
        i = int(row['Node A']) -1
        j = int(row['Node B']) -1
        new = [row['weights'], row['times']]

        #Check if the existing value is smaller
        if adj_matrix[i][j][0] == 0:
            adj_matrix[i][j] = new
        elif new[0] < adj_matrix[i][j][0]:
            adj_matrix[i][j] = new

    return adj_matrix

def dijkstra(adj_matrix, start):
    # Initialize weights, times and paths
    num_nodes = len(adj_matrix)
    weight_sums = [float('inf')] * num_nodes
    weight_sums[start] = 0
    times = [float('inf')] * num_nodes
    times[start] = 0

    # Initialize priority queue
    pq = [(0, 0, start)]
    path_list = [[i] for i in range(num_nodes)]

    # Implementation of Dijkstra's algorithm
    while pq:
        # Pop the node with the minimum weight
        current_weight, current_time, current_node = heapq.heappop(pq)
        # Check if the current node is already processed
        if current_weight > weight_sums[current_node]:
            continue

        for neighbor, weight in enumerate(adj_matrix[current_node]):
            # check if there is an edge
            if weight[0] > 0:
                # Calculate new weights and times
                w = current_weight + weight[0]
                t = current_time + weight[1]

                # Update weights and times if a shorter path is found
                if w < weight_sums[neighbor]:
                    weight_sums[neighbor] = w
                    times[neighbor] = t
                    path_list[neighbor] = path_list[current_node] + list([neighbor])
                    heapq.heappush(pq, (w, t, neighbor))

    return weight_sums, times, path_list

def dijkstra_component(adj_matrix):
  #Call Dijkstra with adjusted values (shifted by 1 due to the adjacency matrix)
  w_s, t, path = dijkstra(mat, 93)
  r = path[161]
  route = [(r[i]+1, r[i+1]+1) for i in range(len(r) - 1)]
  return t[161], route

def visual(route):
  #Prepare necessary dataframes
  pos_data = pd.read_csv("/content/drive/MyDrive/ProO/data/graph_cleaned_weighted.csv")
  pos_data = pos_data.drop_duplicates()
  df_vis = pd.DataFrame()
  df_vis['Nodes'] = list(pos_data['Node A']) + list(pos_data['Node B'])
  df_vis['X'] = list(pos_data['X of A']) + list(pos_data['X of B'])
  df_vis['Y'] = list(pos_data['Y of A']) + list(pos_data['Y of B'])
  df_nodes = df_vis.drop_duplicates()
  graph = nx.DiGraph()


  #Add nodes with their positions
  for index, row in df_nodes.iterrows():
      graph.add_node(row['Nodes'], pos=(row['X'] / 10, row['Y'] / 10))

  pos = nx.get_node_attributes(graph, 'pos')

  for edge in route:
      graph.add_edge(edge[0], edge[1])

  #Plot the graph
  plt.figure(figsize=(20, 16))
  pos = nx.get_node_attributes(graph, 'pos')
  nx.draw_networkx_nodes(graph, pos=pos, node_size=30, node_color='blue')
  nx.draw_networkx_edges(graph, pos)
  plt.title('Shortest path for this cluster')
  plt.show()


#graph_times = pd.read_csv("/content/drive/MyDrive/ProO/data/test.csv")

#t = prep(graph_times)
#mat = adjacency(t)
#time, route = dijkstra_component(mat)
#print(time)
#visual(route)
