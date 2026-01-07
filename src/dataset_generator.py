import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_dataset():
  """
  Generates time series dataset according to specifications.

  Returns:
    pd.DataFrame: DataFrame with columns 'timestamp', 'value', and 'is_anomaly'.
  """
  np.random.seed(42)      # Set seed to ensure reproducibility

  # For testing purposes, generate a datapoint every 5mins for 3 days
  # 1 point/5mins * 60 mins/hr * 24hr/day * 3days
  n_points = int((60/5) * 24 * 3)

  # Generate the timestamp every 5mins
  timestamps = pd.date_range(start='2024-01-01', periods=n_points, freq='5min')

  # Define seasonality
  amplitude = 50
  hour_in_decimal = timestamps.hour + timestamps.minute / 60 # Converts time to decimal representation (e.g 10:30 will become 10.5, 4:15 will become 4.25)
  seasonality = amplitude * np.abs(np.sin(np.pi * hour_in_decimal / 24))
  noise = np.random.normal(0, 1, len(timestamps))
  values = (seasonality + noise).values

  # Define anomalies
  # Point Anomalies
  spike_indices = [50, 150, 300, 450, 500]
  drop_indices = [100, 200, 350]
  values[spike_indices] += 10
  values[drop_indices] -= 10

  # Contextual Anomalies
  values[250:280] = 0   # Sudden data/log loss
  values[550:600] += 20 # Sustained increased traffic

  # Initialize the dataframe
  synthetic_ts = pd.DataFrame({'timestamp': timestamps, 'value': values})

  # Add labels to anomaly points
  synthetic_ts['is_anomaly'] = 0
  synthetic_ts.loc[spike_indices, 'is_anomaly'] = 1
  synthetic_ts.loc[drop_indices, 'is_anomaly'] = 1
  synthetic_ts.loc[250:280, 'is_anomaly'] = 1
  synthetic_ts.loc[550:600, 'is_anomaly'] = 1

  # To better represent web traffic, get the floor(or ceiling) value
  synthetic_ts['value'] = synthetic_ts['value'].apply(math.floor)

  return synthetic_ts