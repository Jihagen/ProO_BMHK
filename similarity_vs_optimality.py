import matplotlib.pyplot as plt
import numpy as np
from knotenpaare_neu import knotenpaare
from Graph_Algorithm import prep, adjacency, dijkstra, dijkstra_component, visual

def similarity_optimality(graph_times):
  normalization = np.linspace(0, 1, 20)
  values = []
  for t in normalization:
    k = knotenpaare(t)
    p = prep(graph_times, k)
    a = adjacency(p)
    time, _ = dijkstra_component(a)
    values += [time * 60]

  plt.plot(normalization, values, marker='o', linestyle='-')
  plt.xlabel('Factor')
  plt.ylabel('Time taken in minutes')
  plt.title('How this similarity component effects the optimality of our shortest path')
  plt.show()
