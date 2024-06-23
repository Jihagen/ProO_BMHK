import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataTransformer:
    def __init__(self, log_transform=False):
        self.scaler = StandardScaler()
        self.log_transform = log_transform
        self.features = None

    def fit_transform(self, data):
        data = data.copy()  # Make a copy to avoid SettingWithCopyWarning
        
        # Extract relevant columns for date and time
        day_column = data['Unnamed: 1045'].astype(int)
        timestamp_column = data['Unnamed: 1047']

        # Parse the timestamp to extract hours and minutes
        timestamp_column = timestamp_column.str.strip()
        timestamp_column = timestamp_column.replace('', '00_00_00')  # Replace empty timestamps with '00_00_00'

        # Split the timestamp into hours, minutes, and seconds
        timestamps_split = timestamp_column.str.split('_', expand=True).astype(int)
        hours = timestamps_split[0]
        minutes = timestamps_split[1]

        # Convert time (hours and minutes) to sine and cosine
        time_in_minutes = hours * 60 + minutes
        time_sin = np.sin(2 * np.pi * time_in_minutes / 1440)  # 1440 minutes in a day
        time_cos = np.cos(2 * np.pi * time_in_minutes / 1440)

        # Assume day_column represents the day of the month and create a mock date
        data.loc[:, 'date'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(day_column - 1, unit='D')
        data.loc[:, 'weekday'] = data['date'].dt.weekday
        data.loc[:, 'is_weekend'] = data['weekday'].apply(lambda x: 1 if x >= 5 else 0)

        # Add transformed columns to the original data
        data.loc[:, 'time_sin'] = time_sin
        data.loc[:, 'time_cos'] = time_cos

        # Select all features except the original cyclic ones, date columns, and the constant column
        features_to_exclude = ['Unnamed: 1045', 'Unnamed: 1046', 'Unnamed: 1047', 'date']
        self.features = data.columns.difference(features_to_exclude)

        # Apply log transformation if needed
        if self.log_transform:
            # Add a small constant to avoid log(0) issues
            data.loc[:, self.features] = data[self.features].apply(lambda x: np.log1p(x - x.min() + 1))

        # Standardize the features
        data_scaled = pd.DataFrame(self.scaler.fit_transform(data[self.features]), columns=self.features, index=data.index)

        return data_scaled, data

    def transform(self, data):
        data = data.copy()  # Make a copy to avoid SettingWithCopyWarning
        
        # Apply the same transformations to new data
        day_column = data['Unnamed: 1045'].astype(int)
        timestamp_column = data['Unnamed: 1047']

        timestamp_column = timestamp_column.str.strip()
        timestamp_column = timestamp_column.replace('', '00_00_00')

        timestamps_split = timestamp_column.str.split('_', expand=True).astype(int)
        hours = timestamps_split[0]
        minutes = timestamps_split[1]

        time_in_minutes = hours * 60 + minutes
        time_sin = np.sin(2 * np.pi * time_in_minutes / 1440)
        time_cos = np.cos(2 * np.pi * time_in_minutes / 1440)

        data.loc[:, 'date'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(day_column - 1, unit='D')
        data.loc[:, 'weekday'] = data['date'].dt.weekday
        data.loc[:, 'is_weekend'] = data['weekday'].apply(lambda x: 1 if x >= 5 else 0)

        data.loc[:, 'time_sin'] = time_sin
        data.loc[:, 'time_cos'] = time_cos

        # Apply log transformation if needed
        if self.log_transform:
            # Add a small constant to avoid log(0) issues
            data.loc[:, self.features] = data[self.features].apply(lambda x: np.log1p(x - x.min() + 1))

        # Standardize the features using the previously fitted scaler
        data_scaled = pd.DataFrame(self.scaler.transform(data[self.features]), columns=self.features, index=data.index)

        return data_scaled, data
