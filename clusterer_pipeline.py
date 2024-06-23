import pickle
import pandas as pd
from clusterer import Clusterer

class ClusteringAlgo:
    def __init__(self, models_path='models'):
        self.transformer, self.first_level_clusterer, self.second_level_clusterers = self.load_models(models_path)

    @staticmethod
    def load_models(path='models'):
        with open(f'{path}/data_transformer.pkl', 'rb') as f:
            transformer = pickle.load(f)
        with open(f'{path}/first_level_clusterer.pkl', 'rb') as f:
            first_level_clusterer = pickle.load(f)
        with open(f'{path}/second_level_clusterers.pkl', 'rb') as f:
            second_level_clusterers = pickle.load(f)
        return transformer, first_level_clusterer, second_level_clusterers

    def process_row(self, row):
        row_scaled, transformed_row = self.transformer.transform(pd.DataFrame([row]))
        
        # Select the columns for time-based clustering
        time_based_features = ['is_weekend', 'time_sin', 'time_cos']
        time_based_data = row_scaled[time_based_features]
        
        # Predict first-level clusters
        first_level_cluster = self.first_level_clusterer.predict(time_based_data)[0]
        transformed_row['time_based_cluster'] = first_level_cluster
        
        # Initialize a list to store second-level predictions
        second_level_cluster = None
        
        # Process the first-level cluster for the row
        if first_level_cluster in self.second_level_clusterers:
            # Drop the unwanted columns for the second-level clustering
            columns_to_exclude = ['is_weekend', 'time_sin', 'time_cos', 'weekday']
            second_level_features = row_scaled.drop(columns=columns_to_exclude)
            
            # Predict second-level clusters for the row
            second_level_clusterer = self.second_level_clusterers[first_level_cluster]
            second_level_cluster = second_level_clusterer.predict(second_level_features)[0]
        
        row['first_level_cluster'] = first_level_cluster
        row['second_level_cluster'] = second_level_cluster
        
        return row

    def dataframe_prediction(self, df):
        # Transform the input dataframe
        data_scaled, transformed_data = self.transformer.transform(df)
        
        # Select the columns for time-based clustering
        time_based_features = ['is_weekend', 'time_sin', 'time_cos']
        time_based_data = data_scaled[time_based_features]
        
        # Predict first-level clusters
        first_level_labels = self.first_level_clusterer.predict(time_based_data)
        
        # Add the first-level cluster labels to the original transformed_data DataFrame
        transformed_data['time_based_cluster'] = first_level_labels
        
        # Initialize a list to store second-level predictions
        all_second_level_labels = []
        
        # Process each first-level cluster in the data
        for first_level_label in transformed_data['time_based_cluster'].unique():
            if first_level_label in self.second_level_clusterers:
                # Get the subset for the current first-level cluster
                second_level_features = data_scaled[data_scaled.index.isin(transformed_data[transformed_data['time_based_cluster'] == first_level_label].index)]
                
                # Drop the unwanted columns for the second-level clustering
                columns_to_exclude = ['is_weekend', 'time_sin', 'time_cos', 'weekday']
                second_level_features = second_level_features.drop(columns=columns_to_exclude)
                
                # Predict second-level clusters for the subset
                second_level_clusterer = self.second_level_clusterers[first_level_label]
                second_level_labels = second_level_clusterer.predict(second_level_features)
                
                # Add the second-level cluster labels to the second_level_features DataFrame
                second_level_features['second_level_cluster'] = second_level_labels
                
                # Store the results
                all_second_level_labels.append(second_level_features[['second_level_cluster']])
        
        # Concatenate all second-level labels
        final_second_level_labels = pd.concat(all_second_level_labels)
        
        # Merge the second-level cluster labels back into the transformed_data DataFrame
        transformed_data = transformed_data.merge(final_second_level_labels, left_index=True, right_index=True, how='left')
        
        return transformed_data

def run(input_data):
    clusterer = ClusteringAlgo()

    if isinstance(input_data, pd.DataFrame):
        return clusterer.dataframe_prediction(input_data)
    elif isinstance(input_data, dict):
        return clusterer.process_row(input_data)
    else:
        raise ValueError("Unsupported input type. Expected DataFrame or dict.")

