import matplotlib.pyplot as plt
import numpy as np
from knotenpaare_neu import knotenpaare
from Graph_Algorithm import prep, adjacency, dijkstra, dijkstra_component, visual_3


def similarity_optimality(graph_times):
  normalization = np.linspace(0, 1, 20)
  values = []
  for t in normalization:
    k = knotenpaare(t)
    p = prep(graph_times,k )
    a = adjacency(p)
    t, _= dijkstra_component(a)
    values.append(t)

  plt.plot(normalization, values, marker='o', linestyle='-')
  plt.xlabel('Factor')
  plt.ylabel('Time taken')
  plt.title('How this similarity component effects the optimality of our shortest path')
  plt.show()



def visualize_sim_difference(graph_times):
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

  visual_3(route_1, route_0, route_05, ("Rot: Similarity auf 1 Zeit: " + str(time_1) + " Blau: Similarity auf 0, Zeit: "+ str(time_0) + " Grün: Similarity auf 0.5" + str(time_05)))


def optimal_time(graph_times, month, code):
  pos_data = pd.read_csv("graph_imp_weighted.csv")
  route_df = pd.read_csv("opt_routes_cleaned.csv")
  l = route_df[route_df["Month"] == month and route_df["Code"]== code]
  r_df = pd.DataFrame({'Edge_Tuples': l})

  pos_data["Node A"] = pos_data["Node A"].astype(int)
  pos_data["Node B"] = pos_data["Node B"].astype(int)
  pos_data["Edge_Tuples"] = list(zip(pos_data['Node A'], pos_data['Node B']))

  merged = pd.merge(r_df, pos_data, on="Edge_Tuples")
  merged = merged.drop(columns=['X of A', 'X of B', 'Y of A', 'Y of B', 'Avg Speed', 'Med Speed'])
  merged_again = pd.merge(merged, graph_times, on="Edge names")
  time = merged_again["times"].sum()
  return time





