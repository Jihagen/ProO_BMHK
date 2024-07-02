import pandas as pd
import warnings
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
    
    '''
    # Example for processing a DataFrame
    transformed_data = run(data)
    print("Transformed Data:")
    print(transformed_data.columns, transformed_data.head())
    print(transformed_data.second_level_cluster.unique())
    transformed_data.to_csv('clustered_data_all.csv', index=False, header=True)
    '''
    # Suppress SettingWithCopyWarning
    pd.options.mode.chained_assignment = None  # default='warn'

    # Suppress PerformanceWarning
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

    # Suppress all warnings (use with caution)
    warnings.filterwarnings('ignore')
   ### Example for processing a single row
    example_row = data.sample(n=1)
    transformed_row = run(example_row)
    print("Transformed Row:")
    print(transformed_row)

    ### Extract Information
    weights = ClusterWeights('clustered_data_all.csv','distance_neu.csv' )
    cluster = weights.generate_cluster_identifier(transformed_row.iloc[0])
    print(cluster)
    graph_times = weights. get_lookup_table(cluster)
    print(graph_times)

    ### Example for calculating a shortest path for a cluster
    t = prep(graph_times)
    mat = adjacency(t)
    time, route = dijkstra_component(mat)
    print("required time for the shortest path: " + str(time*60) + " minutes")
   # visual(route,cluster)

   # print(example_row)
    # print(transformed_row)
