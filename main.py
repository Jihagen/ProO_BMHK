import pandas as pd
from clusterer_pipeline import ClusteringAlgo, run
from data_transformer import DataTransformer
from clusterer import Clusterer
from weights import ClusterWeights
from Graph_Algorithm import prep, adjacency, dijkstra, dijkstra_component, visual


# Example usage
if __name__ == "__main__":
    # Load data, Replace with actual input data! 

    ### for processing data-all once!

    data = pd.read_csv('data-all.csv')

    # Initialize the clusterer and perform predictions
    clusterer = ClusteringAlgo()
    '''
    # Example for processing a DataFrame
    transformed_data = run(data)
    print("Transformed Data:")
    print(transformed_data.columns, transformed_data.head())
    transformed_data.to_csv('clustered_data_all.csv', index=False, header=True)
    '''
   ### Example for processing a single row
    example_row = data.iloc[0].to_dict()
    transformed_row = run(example_row)
    print("Transformed Row:")
    print(transformed_row)

    ### Extract Information
    weights = ClusterWeights('clustered_data_all.csv','distance_neu.csv' )
    cluster = weights.generate_cluster_identifier(transformed_row)
    print(cluster)
    graph_times = weights. get_lookup_table(cluster)
    print(graph_times)

    ### Example for calculating a shortest path for a cluster
    t = prep(graph_times)
    mat = adjacency(t)
    time, route = dijkstra_component(mat)
    print("required time for the shortest path: " + str(time))
    visual(route)
   
