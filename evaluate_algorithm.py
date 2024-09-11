import pandas as pd
import warnings
from clusterer_pipeline import ClusteringAlgo, run
from data_transformer import DataTransformer
from clusterer import Clusterer
from weights import ClusterWeights
from Graph_Algorithm import prep, adjacency, dijkstra, dijkstra_component, visual, visual_2
from knotenpaare_neu import knotenpaare
from similarity_vs_optimality import visualize_sim_difference, similarity_optimality, optimal_time, visualize_frequency
import matplotlib.pyplot as plt


def evaluate_algorithm(data, num_samples=10):
    together_edges_list = []
    routes_list = []
    times_list = []
    counter = 0
    num_samples = min(num_samples, len(data))
    #edge_count = pd.DataFrame()
    #edge_count["Pairs"] = [(0, 0)]
    #edge_count["Counter"] = [0]

    
    for i in range(num_samples):
        # Select a random row
        example_row = data.iloc[[i]]
        
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

        #for i in opt_route:
            #if i in edge_count["Pairs"].values:
                #edge_count.loc[edge_count['Pairs'] == i, 'Counter'] += 1
            #else:
                #edge_count = pd.concat([edge_count, pd.DataFrame({'Pairs': [i], 'Counter': [1]})], ignore_index=True)
        if month == "May"


        # Collect together_edges
        together_edges_list.append(together_edges)
        #routes_list.append(route)
        times_list.append(time)
        counter += 1

    #bin_edges = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    plt.hist(together_edges_list, bins=10, edgecolor='black')
    print(together_edges_list)

    #visualize_frequency(edge_count)

    # Titel und Achsenbeschriftungen hinzufügen
    plt.title('Histogramm Evaluation')
    plt.xlabel('Werte')
    plt.ylabel('Häufigkeit')
    plt.show()

    # Calculate average together_edges
    avg_together_edges = sum(together_edges_list) / counter
    avg_time = sum(times_list) / counter


    return counter, avg_together_edges, avg_time
