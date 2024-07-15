import pandas as pd
import warnings
from clusterer_pipeline import ClusteringAlgo, run
from data_transformer import DataTransformer
from clusterer import Clusterer
from weights import ClusterWeights
from Graph_Algorithm import prep, adjacency, dijkstra, dijkstra_component, visual, visual_2
from knotenpaare_neu import knotenpaare
from similarity_vs_optimality import visualize_sim_difference, similarity_optimality, optimal_time

def evaluate_algorithm(data, num_samples=10):
    together_edges_list = []
    counter = 0
    
    for _ in range(num_samples):
        # Select a random row
        example_row = data.sample(n=1)
        
        # Transform the row
        transformed_row = run(example_row)
        
        # Extract relevant information
        weekday = transformed_row['weekday'].iloc[0]
        time_str = transformed_row['Unnamed: 1047'].iloc[0]
        time_formatted = f"{time_str.split('_')[0]}:{time_str.split('_')[1]}"
        month = transformed_row['Unnamed: 1046'].iloc[0]
        
        # Generate cluster and lookup table
        weights = ClusterWeights('clustered_data_all.csv', 'distance_neu.csv')
        cluster = weights.generate_cluster_identifier(transformed_row.iloc[0])
        graph_times = weights.get_lookup_table(cluster)

        # Calculate similarity factor
        fac = knotenpaare(0.5)

        # Calculate shortest path for a cluster
        t = prep(graph_times, fac)
        mat = adjacency(t)
        time, route = dijkstra_component(mat)
        time = time * 60
        
        # Calculate optimal path
        opt_route, together_edges, opt_time = optimal_time(graph_times, month, time_str, route)
        if (opt_route, together_edges, opt_time) == (0, 0, 0):
            continue

        # Collect together_edges
        together_edges_list.append(together_edges)
        counter += 1

    # Calculate average together_edges
    avg_together_edges = sum(together_edges_list) / counter
    
    return counter, avg_together_edges
