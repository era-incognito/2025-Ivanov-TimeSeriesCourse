import numpy as np


def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """Calculate the Euclidean distance between two time series."""

    return float(np.linalg.norm(ts1 - ts2))

def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """Calculate the normalized Euclidean distance.

    Time series are z‑normalized independently and then the standard
    Euclidean distance is computed between the normalized vectors.
    """

    ts1_z = (ts1 - np.mean(ts1)) / np.std(ts1)
    ts2_z = (ts2 - np.mean(ts2)) / np.std(ts2)

    return ED_distance(ts1_z, ts2_z)


def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1) -> float:
    """
    Calculate DTW distance

    Parameters
    ----------
    ts1: first time series
    ts2: second time series
    r: warping window size
    
    Returns
    -------
    dtw_dist: DTW distance between ts1 and ts2
    """

    n = len(ts1)
    m = len(ts2)

    # Initialize the cost matrix
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    # Apply the warping window size (Sakoe‑Chiba band)
    window = max(int(r * max(n, m)), 1)

    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m, i + window)
        for j in range(j_start, j_end + 1):
            cost = (ts1[i - 1] - ts2[j - 1]) ** 2
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],  # insertion
                dtw_matrix[i, j - 1],  # deletion
                dtw_matrix[i - 1, j - 1]  # match
            )

    return dtw_matrix[n, m]
