import rrcf

def fit(series,
  threshold=20,
  num_trees=100,
  tree_size=100):
  """
  Perform anomaly detection using Robust Random Cut Forest

  Parameters:
    series: pd.Series, Series to analyze
    threshold: int, Threshold to classify as anomaly
    num_trees: int, Number of Trees
    tree_size: int, Tree Size

  Returns:
    anomalies: list[dict] The list of records to process.
            Each dictionary should have the following keys:
            *   `index` (int): Index where the anomaly was found
            *   `score` (int): RRCF CoDisplacement Score.
  """

  anomalies = []

  # Initialize Tree

  forest = [rrcf.RCTree() for _ in range(num_trees)]

  for i, point in enumerate(series):
    avg_codisp = 0 # Initialize Co-Displacement

    for tree in forest:
      # If tree is full, forget the oldest point (Sliding Window)
      if len(tree.leaves) > tree_size:
        tree.forget_point(i - tree_size)

      # Insert the new point and calculate "Co-displacement" (anomaly score)
      tree.insert_point(point, index=i)
      avg_codisp += tree.codisp(i) / num_trees
    
    if avg_codisp > threshold:  # Your chosen threshold
      anomalies.append({"index": i, "score": avg_codisp})

  return anomalies