import pandas as pd
import numpy as np

class ClusterWeights:
    def __init__(self, cluster_data_path, distance_data_path):
        # Load the clustered data and distance data
        self.clustered_data_all = pd.read_csv(cluster_data_path)
        self.distance_data = pd.read_csv(distance_data_path, 
                                         names=['Edge name', 'first node', 'second node', 'X of A', 'X of B', 'Y of A', 'Y of B',
                                                'avg speed', 'median', 'distance'])
        # Create unique clusters
        self.clustered_data_all['cluster'] = self.clustered_data_all['time_based_cluster'].astype(str) + self.clustered_data_all['second_level_cluster'].astype(str)
        self.unique_entries_list = self.clustered_data_all['cluster'].unique().tolist()
        
        # Generate lookup tables
        self.lookup_tables = self._create_lookup_tables()
    
    def _create_lookup_tables(self):
        lookup_tables = {}
        for cluster in self.unique_entries_list:
            current_cluster = self.clustered_data_all[self.clustered_data_all['cluster'] == cluster]
            table_name = 'cluster_' + cluster
            lookup_tables[table_name] = self.weights(current_cluster)
        return lookup_tables
    
    def weights(self, cluster):
        # Step 1: mean per edge in the cluster
        columns_to_exclude = ['Unnamed: 1045', 'Unnamed: 1046', 'Unnamed: 1047', 'date', 'weekday', 'is_weekend',
                              'time_sin', 'time_cos', 'time_based_cluster', 'second_level_cluster', 'cluster']
        remaining_columns = cluster.columns.difference(columns_to_exclude)
        mean_values = cluster[remaining_columns].mean()
        cluster.loc['Average Speed'] = mean_values

        # Step 2: Complete edge weights
        existing_edge_names = set(cluster.columns.difference(columns_to_exclude).tolist())
        all_edge_names = set(self.distance_data['Edge name'].tolist())
        missing_edges = list(all_edge_names - existing_edge_names)
        duplicate_edges = [element for element in missing_edges if '_' in element]
        edges_without_speed = list(set(missing_edges) - set(duplicate_edges))
        
        for edge in edges_without_speed:
            cluster[edge] = np.nan
        
        for edge in duplicate_edges:
            partner_edge = edge.split('_')[0]
            cluster[edge] = 'NaN'
            cluster.loc['Average Speed', edge] = cluster.loc['Average Speed', partner_edge]
        
        avg = cluster.loc['Average Speed'].mean()
        
        for edge in edges_without_speed:
            cluster.loc['Average Speed', edge] = avg
        
        for edge in duplicate_edges:
            if pd.isna(cluster.loc['Average Speed', edge]):
                cluster.loc['Average Speed', edge] = avg

        # Step 3: Joining distance on edge name
        distances = self.distance_data.set_index('Edge name')['distance']
        new_row = distances.reindex(cluster.columns).to_frame().T
        new_row.index = ['distance']
        test_cluster = pd.concat([cluster, new_row])

        # Step 4: Various conversions for the correct time
        km_to_miles = 0.621371
        test_cluster.loc['distance_miles'] = test_cluster.loc['distance'] * km_to_miles
        speeds = test_cluster.loc['Average Speed']
        dists = test_cluster.loc['distance_miles']
        time = dists / speeds
        time.name = 'Time'

        # Step 5: Adding the new row for time to test_cluster
        test_cluster = pd.concat([test_cluster, time.to_frame().T])
        edge_names = test_cluster.columns.tolist()
        times = test_cluster.loc['Time'].tolist()
        edges_time_df = pd.DataFrame({'edge_names': edge_names, 'times': times})
        return edges_time_df
    
    def get_lookup_table(self, cluster):
        table_name = 'cluster_' + cluster
        return self.lookup_tables.get(table_name, None)
    
    def select_cluster_rows(self, cluster):
        current_cluster = self.clustered_data_all[self.clustered_data_all['cluster'] == cluster]
        return current_cluster
    
    def create_dynamic_cluster(self, cluster, prediction): 
        table = self.select_cluster_rows(cluster)
        table.loc[30000] = prediction  # given row is added with index 30000, which is not used in any cluster
        table['Row_average'] = table.iloc[:, :-10].mean(axis=1)  # take mean of row, ignoring those columns that were only relevant for clustering
        print(table)
        table_sorted = table.sort_values(by='Row_average')  # sort table
        prediction_index = 30000
        prediction_position = table_sorted.index.get_loc(prediction_index)  # at which position is prediction row?
        length = len(table_sorted) - 1
        ten_percent = int(np.ceil(0.1 * length))
        half_region_size = ten_percent
        
        # Is our row in the margin area (are there fewer than 10% rows above/below)?
        upper_margin = (ten_percent >= prediction_position)
        lower_margin = (length - ten_percent <= prediction_position)
        if upper_margin or lower_margin:
            # Margin area:
            half_region_size = min(prediction_position, abs(length - prediction_position))  # the region now only consists of +- rows to the closer margin
        start_index = prediction_position - half_region_size
        end_index = prediction_position + half_region_size
        dynamic_cluster = table_sorted.iloc[start_index:end_index + 1]
        print(dynamic_cluster)
        return dynamic_cluster # select header row and desired rows
    
    def weights_for_dynamic_cluster(self, cluster, prediction):
        dynamic_table = self.create_dynamic_cluster(cluster, prediction)
        edges_time_df = self.weights(dynamic_table)
        return edges_time_df
    
    @staticmethod
    def generate_cluster_identifier(row):
        """
        Generate cluster identifier based on time_based_cluster and second_level_cluster information from a DataFrame row.
        """
        return str(row['time_based_cluster']) + str(row['second_level_cluster'])


'''
## Example Usage:
weights = ClusterWeights('clustered_data_all.csv','distance_neu.csv' )
print(weights.get_lookup_table('31'))
'''

''' für den interessierten Leser:
    CLUSTER SCHÖN ANSCHAUEN:
    for key, value in lookup_tables.items():
    print(f"Key: {key}")
    print(value)
    print("-----------------")
'''

'''How to: diese Datei benutzen:
    wähle eines der 12 cluster: ['31', '30', '32', '01', '00', '02', '10', '11', '21', '22', '20', '12']
    der Aufruf lookup('31') gibt dann ein pd.df mit den spalten 'edge_names' und 'times', welches weiterverwendet werden soll
'''

'''ACHTUNG:
    diese file importiert (in zeile 14 und 70) die datein distance_neu.csv und clustered_data_all.csv. 
    Wenn diese nicht im selben Verzeichnis liegen, muss der import pfad angepasst werden!
'''
