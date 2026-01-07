from sklearn.cluster import DBSCAN

def fit(data,
  eps=0.5, 
  min_samples=4):
  """
  Perform DBSCAN Clustering

  Parameters:
    `data`: Data to perform clustering on
    `eps`: float, Epsilon Hyperparameter
    `min_samples`: int, Minimum Samples to be called a cluster

  Returns:
    `clusters`: dict, Cluster attributes
       * `clusters`: Cluster Attributes
       * `n_clusters`: Number of Clusters 
       * `n_noise`: Number of noise points
  """
  dbscan = DBSCAN(eps=eps, min_samples=min_samples)
  computed_clusters = dbscan.fit_predict(data)

  n_clusters = len(set(computed_clusters)) - (1 if -1 in computed_clusters else 0)
  n_noise = list(computed_clusters).count(-1)

  print(f"DBSCAN Params | EPS: {eps}, min_sample: {min_samples}")
  print(f"* Cluster Count: {n_clusters}")
  print(f"* Noise Points: {n_noise}")

  return {
      'clusters': computed_clusters,
      'n_clusters': n_clusters,
      'n_noise': n_noise
  }