import pandas as pd
import warnings
from clusterer_pipeline import ClusteringAlgo, run
from data_transformer import DataTransformer
from clusterer import Clusterer
from weights import ClusterWeights
from Graph_Algorithm import prep, adjacency, dijkstra, dijkstra_component, visual, visual_2
from knotenpaare_neu import knotenpaare
from similarity_vs_optimality import similarity_optimality, visualize_sim_difference, optimal_time
from evaluate_algorithm import evaluate_algorithm

# Example usage
if __name__ == "__main__":
   

    data = pd.read_csv('test_split.csv')
    example_row1 = data.sample(n=1)
    example_row2 = data.sample(n=1)
    print(example_row1)
    print(example_row2)
    # Initialize the clusterer and perform predictions
    
    '''
    # Example for processing a DataFrame
    ### change data to data-all.csv to process 
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
    transformed_row = run(example_row1)
    print("Transformed Row:")
    print(transformed_row)
    weekday = transformed_row['weekday'].iloc[0]
    time_str = transformed_row['Unnamed: 1047'].iloc[0]
    time_formatted = f"{time_str.split('_')[0]}:{time_str.split('_')[1]}"
    month = transformed_row['Unnamed: 1046'].iloc[0]
    
    weekday_dict = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday"
    }
    formatted_weekday = weekday_dict[weekday]



    ### Extract Information
    weights = ClusterWeights('clustered_data_all.csv','distance_neu.csv' )
    cluster = weights.generate_cluster_identifier(transformed_row.iloc[0])
    #graph_times = weights.get_lookup_table(cluster)

    graph_times = weights.weights_for_dynamic_cluster(cluster, transformed_row.iloc[0])

    

    #Calculate similarity factor
    fac = knotenpaare(1)

    ### Example for calculating a shortest path for a cluster
    t = prep(graph_times, fac)
    mat = adjacency(t)
    time, route = dijkstra_component(mat)
    print(f"Identified {formatted_weekday} at {time_formatted} as cluster: {cluster}")
    print("required time for the shortest path: " + str(time*60) + " minutes")
    time = time * 60
    visual(route,cluster)

    similarity_optimality(graph_times)
    visual(route,cluster)
    visualize_sim_difference(graph_times)

    opt_route, together_edges, opt_time = optimal_time(graph_times, month, time_str, route)
    time_difference = opt_time - time

    print("Time difference:" + str(time_difference) + " minutes")
    print("Gemeinsame Kanten geteilt durch gefahrene Kanten: " + str(together_edges))
    visual_2(route, opt_route, "path taken: green, opt_path: orange")

    c, av = evaluate_algorithm(data, 200)
    print(c)
    print(av)
