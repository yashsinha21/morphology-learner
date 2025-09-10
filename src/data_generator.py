import pandas as pd
import numpy as np

def generate_data_from_csv(csv_file, accuracy_rate, n):
    """
    Generate a data set from a paradigm CSV with a given accuracy rate.
  
    Parameters
    ----------
    csv_file : str
        Path to the CSV file containing the paradigm. Each row of the file
        (except the first) contains the specifications of a 'cell' of the 
        paradigm: the binary feature specification (0 or 1) for that cell, 
        and the morph associated with that cell. The morph is in the last column,
        while the features are in the previous columns. The first row contains
        the names of the features.
    accuracy_rate : float
        Probability [0,1] that the morph corresponding to a feature in the data 
        is picked from the correct cell.
    n : int
        Number of samples to generate.

    Returns
    -------
    feature_names : list of str
        Names of the features (from the first row, all columns except the last)
    morphs : list of str
        List of all unique morphs in the paradigm.
    data_features : np.ndarray
        2D array of shape (n X feature_count) that stores the features of each sample.
    data_morph : np.ndarray
        1D array of shape (n) that stores the morph of each sample.
    error_rate : float
        The proportion of incorrect morphs. 
    """
    # Load CSV without treating first row as header
    df = pd.read_csv(csv_file, header=None)
    
    # Extract feature names from the first row (all except last column)
    feature_names = df.iloc[0, :-1].tolist()
    
    # Extract paradigm data (all rows except first)
    paradigm_features = df.iloc[1:, :-1].to_numpy(dtype=int)
    paradigm_morph = df.iloc[1:, -1].to_numpy()

    # Extract all unique morphs
    morph_list = list(np.unique(paradigm_morph))

    # Number of cells and features
    cell_count, feature_count = paradigm_features.shape

    # Initialize generated data
    data_features = np.empty((n, feature_count), int)
    data_morph = np.empty((n), dtype=object)
    incorrect_morphs = 0

    # Generate n samples
    for i in range(n):
        # Randomly pick a cell index
        cell_idx = np.random.randint(cell_count)

        # Copy feature specification
        data_features[i] = paradigm_features[cell_idx]
  

        # Pick morph based on accuracy_rate
        correct_morph = paradigm_morph[cell_idx]
        if np.random.rand() < accuracy_rate:
            data_morph[i] = correct_morph
        else:
            # Pick a different morph and update incorrect morph count. 
            data_morph[i] = np.random.choice(
                [morph for morph in morph_list if morph != correct_morph])
            incorrect_morphs += 1

    # Compute proportion of incorrect morphs
    error_rate = incorrect_morphs / n

    return feature_names, morph_list, data_features, data_morph, error_rate
