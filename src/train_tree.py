from sklearn.tree import DecisionTreeClassifier

def train_tree(X, y, min_imp_dec):
    """
    Trains a Decision Tree on the given dataset. 

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (sample_count X feature_count)
    y : np.ndarray
        Target labels (morphemes)
    min_imp_dec : float
        The minimum impurity decrease required to split a node.

    Returns
    -------
    clf : DecisionTreeClassifier
        Trained decision tree model
    """

    # Initialize the decision tree classifier  
    clf = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=None,            
        min_samples_leaf=1,        
        random_state=1,
        min_impurity_decrease=min_imp_dec        
    )

    # Train the model on the training set
    clf.fit(X_train, y_train)

    #Return the trained model for further use
    return clf
