import pandas as pd
import numpy as np
import csv
from collections import defaultdict

# Pfad zur Datei

route_all = pd.read_csv("route-all.csv", names = range(100))

def knotenpaare(t):
  # Path to the file
  file_path = 'route-all.csv'

  # Dictionary to store the edges and their counters
  edge_count = defaultdict(int)
  last_row = None

  # Read the file and extract node pairs
  with open(file_path, newline='') as csvfile:
      reader = csv.reader(csvfile)
      for row in reader:
          # Extract only the relevant nodes (excluding the last two entries)
          nodes = row[:-2]
          last_row = row
          # Iterate through the node pairs
          for i in range(len(nodes) - 1):
              edge = (nodes[i], nodes[i + 1])
              edge_count[edge] += 1

  #print(f"Number of unique edges: {len(edge_count)}")
  #print(f"Lat row: {last_row}")
  # Normalize the edges
  normalized = {edge: 1 - ((count / 3212)*t) for edge, count in edge_count.items()}

  #t[0,1]
  # Output the edges and their counters in a list for better representation
  edges_normalized = list(normalized.items())
  edges_normalized  # display the entries for verification
  edges = []
  knoten1 = []
  knoten2 = []
  zähler= []
  zähler_normalisiert = []

  for edge, count in edge_count.items():
      knoten1.append(edge[0])
      knoten2.append(edge[1])
      zähler.append(count)
      zähler_normalisiert.append(normalized[edge])

  # Create DataFrames
  normalized_df = pd.DataFrame({
      'Knoten1': knoten1,
      'Knoten2': knoten2,
      'Zähler': zähler,
      'Zähler_normalisiert': zähler_normalisiert
  })
  return normalized_df
  # Save DataFrames as CSV
  #output_file_path = '/content/drive/MyDrive/ProO/data/Knotenpaare_normalisiert.csv'
  '''
  output_file_path = 'Knotenpaare_normalisiert.csv'
  normalized_df.to_csv(output_file_path, index=False)

  print(f'Die normalisierten Knotenpaare wurden erfolgreich in {output_file_path} gespeichert.')
'''
#hier werte zwischen 0 und 1 eingeben:
knotenpaare(0.2)
