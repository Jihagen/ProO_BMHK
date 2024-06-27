import pandas as pd
from clusterer_pipeline import ClusteringAlgo, run
from data_transformer import DataTransformer
from clusterer import Clusterer

# Example usage
if __name__ == "__main__":
    # Load data, Replace with actual input data! 

    ### for processing data-all once!

    ### data = pd.read_csv('data-all.csv')

    # Initialize the clusterer and perform predictions
    clusterer = ClusteringAlgo()

    # Example for processing a DataFrame
    transformed_data = run(data)
    print("Transformed Data:")
    print(transformed_data.head())

    # transformed_data.to_csv("clustered_data_all.csv", index = False)

    ### Example for processing a single row
    example_row = data.iloc[0].to_dict()
    transformed_row = run(example_row)
    print("Transformed Row:")
    print(transformed_row)
