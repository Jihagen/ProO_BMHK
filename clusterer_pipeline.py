import pickle
import pandas as pd
from clusterer import Clusterer
from data_transformer import DataTransformer

class ClusteringAlgo:
    def __init__(self, models_path='models', train_split_path ='train_split.csv'):
        self.transformer, self.first_level_clusterer, self.second_level_clusterers = self.load_models(models_path)
        self.train_data = pd.read_csv(train_split_path)

    @staticmethod
    def load_models(path='models'):
        with open(f'{path}/data_transformer(1).pkl', 'rb') as f:
            transformer = pickle.load(f)
        with open(f'{path}/first_level_clusterer(1).pkl', 'rb') as f:
            first_level_clusterer = pickle.load(f)
        with open(f'{path}/second_level_clusterers(1).pkl', 'rb') as f:
            second_level_clusterers = pickle.load(f)#
        print("Successfully loaded models. ")
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
        
        row['time_based_cluster'] = first_level_cluster
        row['second_level_cluster'] = second_level_cluster
        
        return row

    def dataframe_prediction(self, df):
        # Combine train data with the new data
        combined_data = pd.concat([df, self.train_data], ignore_index=True)
        print(combined_data.head())
        # Transform the combined data
        data_scaled, transformed_combined_data = self.transformer.transform(combined_data)
        
        # Select the columns for time-based clustering
        time_based_features = ['is_weekend', 'time_sin', 'time_cos']
        time_based_data = data_scaled[time_based_features]

        # Predict first-level clusters
        first_level_labels = self.first_level_clusterer.predict(time_based_data)
        transformed_combined_data['time_based_cluster'] = first_level_labels

        # Initialize a list to store second-level predictions
        all_second_level_labels = []

        # Process each first-level cluster in the data
        for first_level_label in transformed_combined_data['time_based_cluster'].unique():
            if first_level_label in self.second_level_clusterers:
                second_level_features = data_scaled[data_scaled.index.isin(
                    transformed_combined_data[transformed_combined_data['time_based_cluster'] == first_level_label].index)]

                columns_to_exclude = ['is_weekend', 'time_sin', 'time_cos', 'weekday']
                second_level_features = second_level_features.drop(columns=columns_to_exclude)

                second_level_clusterer = self.second_level_clusterers[first_level_label]
                second_level_labels = second_level_clusterer.predict(second_level_features)
                second_level_features['second_level_cluster'] = second_level_labels
                all_second_level_labels.append(second_level_features[['second_level_cluster']])

        final_second_level_labels = pd.concat(all_second_level_labels)
        transformed_combined_data = transformed_combined_data.merge(final_second_level_labels, left_index=True, right_index=True, how='left')
 
        return transformed_combined_data.head(1)

def run(input_data):
    clusterer = ClusteringAlgo()

    if isinstance(input_data, pd.DataFrame):
        return clusterer.dataframe_prediction(input_data)
    else:
        raise ValueError("Unsupported input type. Expected DataFrame or dict.")

