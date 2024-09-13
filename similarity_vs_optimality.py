import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from knotenpaare_neu import knotenpaare
from Graph_Algorithm import prep, adjacency, dijkstra, dijkstra_component, visual_3
import csv
import networkx as nx

def similarity_optimality(graph_times):
  normalization = np.linspace(0, 1, 20)
  values = []

  #Calculate path times
  for t in normalization:
    k = knotenpaare(t)
    p = prep(graph_times,k )
    a = adjacency(p)
    t, _= dijkstra_component(a)
    values.append(t)

  #Plot results
  plt.plot(normalization, values, marker='o', linestyle='-')
  plt.xlabel('Factor')
  plt.ylabel('Time taken')
  plt.title('How this similarity component effects the optimality of our shortest path')
  plt.show()



def visualize_sim_difference(graph_times):
  #Calculate all 3 routes
  k_1 = knotenpaare(1)
  p_1 = prep(graph_times, k_1)
  a_1 = adjacency(p_1)
  time_1, route_1 = dijkstra_component(a_1)

  k_0 = knotenpaare(0)
  p_0 = prep(graph_times, k_0)
  a_0 = adjacency(p_0)
  time_0, route_0 = dijkstra_component(a_0)

  k_05 = knotenpaare(0.5)
  p_05 = prep(graph_times, k_05)
  a_05 = adjacency(p_05)
  time_05, route_05 = dijkstra_component(a_05)

  #Plot the routes
  visual_3(route_1, route_0, route_05, ("Rot: Similarity auf 1 Zeit: " + str(time_1) + " Blau: Similarity auf 0, Zeit: "+ str(time_0) + " Grün: Similarity auf 0.5" + str(time_05)))


def optimal_time(graph_times, month, code, path):
  # Remove whitespace from 'code' and 'month'
  code = code.strip()
  month = month.strip()

  # Load the graph data from CSV
  pos_data = pd.read_csv("graph_imp_weighted.csv")
  file_path = "route-all-missing-last-day.csv"

  # Initialize lists to store edges, codes, and months
  result = []
  code_l = []
  month_l = []

  # Read the route file and extract edge tuples, codes, and months
  with open(file_path, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
      temp_list = []
      for entry in row:
        temp_list.append(entry)

      a = [int(temp_list[i]) - 1 for i in range(len(temp_list) - 2)]
      l = [(a[i] + 1, a[i + 1] + 1) for i in range(len(a) - 1)]
      result.append(l)
      code_l.append(temp_list[-1].strip())
      month_l.append(temp_list[-2].strip())

  # Create a DataFrame with edges, months, and codes
  route_df = pd.DataFrame({'Edge_Tuples': result, 'Month': month_l, 'Code': code_l}

  # Filter the DataFrame for the matching month and code
  row_df = route_df[(route_df["Month"] == month) & (route_df["Code"] == code)]["Edge_Tuples"]
  if row_df.empty:
    return 0, 0, 0

  # Get the first matching row and convert it to a list
  row = row_df.iloc[0]
  row = list(row)

  # Create a DataFrame for the matching edge tuples
  r_df = pd.DataFrame({'Edge_Tuples': row})

  # Convert columns to integers and create an 'Edge_Tuples' column in pos_data
  pos_data["Node A"] = pos_data["Node A"].astype(int)
  pos_data["Node B"] = pos_data["Node B"].astype(int)
  pos_data["Edge_Tuples"] = list(zip(pos_data['Node A'], pos_data['Node B']))

  # Merge the DataFrames to match edges with their data
  merged = pd.merge(r_df, pos_data, on="Edge_Tuples")
  merged = merged.drop(columns=['X of A', 'X of B', 'Y of A', 'Y of B', 'Avg Speed', 'Med Speed'])

  # Merge with graph_times to get the times for each edge
  merged_again = pd.merge(merged, graph_times, on="Edge name")

  # Calculate total time in seconds
  time = merged_again["times"].sum() * 60

  # Count how many edges in the row are in the given path
  counter = 0
  for i in row:
    if i in path:
      counter+= 1

  # Calculate the ratio of matching edges
  relation = counter / len(row)


  return row, relation, time


def visualize_frequency(factors):
  #Import necessary data
  pos_data = pd.read_csv("graph_imp_weighted.csv")
  pos_data = pos_data.drop_duplicates()
  df_vis = pd.DataFrame()
  df_vis['Nodes'] = list(pos_data['Node A']) + list(pos_data['Node B'])
  df_vis['X'] = list(pos_data['X of A']) + list(pos_data['X of B'])
  df_vis['Y'] = list(pos_data['Y of A']) + list(pos_data['Y of B'])
  df_nodes = df_vis.drop_duplicates()

  graph = nx.Graph()
  # Add nodes with their positions
  for index, row in df_nodes.iterrows():
    graph.add_node(row['Nodes'], pos=(row['X'] / 10, row['Y'] / 10))

  df_edges = pd.DataFrame()
  df_edges['Node A'] = pos_data['Node A']
  df_edges['Node B'] = pos_data['Node B']
  df_edges['Pairs'] = list(zip(pos_data['Node A'], pos_data['Node B']))
  df_edges = pd.merge(df_edges, factors, on="Pairs")

  df_edges['Width'] = [1] * len(df_edges['Pairs'])
  df_edges.drop_duplicates()

  #Change Width based on frequency of edge
  df_edges["Width"] = df_edges["Counter"] / 100

  graph.add_edges_from(df_edges['Pairs'])

  # Plot the graph
  plt.figure(figsize=(20, 16))
  pos = nx.get_node_attributes(graph, 'pos')
  nx.draw_networkx_nodes(graph, pos=pos, node_size=30, node_color='black')

  nx.draw_networkx_edges(graph, pos=pos, edgelist=df_edges['Pairs'].values, edge_color="black",
                         width=df_edges['Width'])

  plt.show()