

def similarity_optimality:
  normalization = np.linspace(0, 1, 20)
  values = []
  for t in normalization:
    k = knotenpaare(t)
    p = prep(k)
    a = adjacency(p)
    t, _, _ = dijkstra_component(a)
    values.append(t)

  plt.plot(normalization, values, marker='o', linestyle='-')
  plt.xlabel('Factor')
  plt.ylabel('Time taken')
  plt.title('How this similarity component effects the optimality of our shortest path')
  plt.show()
 
