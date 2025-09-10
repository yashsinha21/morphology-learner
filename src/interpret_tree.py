from sklearn.tree import _tree
import numpy as np

def interpret_tree(clf, feature_names):
  """
    Converts a decision tree into linguist-readable morphological specifications
    by extracting paths to each leaf node. 

    Parameters
    ----------
    clf : DecisionTreeClassifier
      A trained decision tree.
    feature_names : list
      List of feature names.

    Returns
    -------
    specifications : list of str
      Linguist-readable rules in the format:[feature specification] <--> morph, 
      where feature specification is a string of features specified with + or -
      (e.g., "+f0, -f1, +f2"). 
    """

  # Access the internal Tree object from the trained classifier
  tree_ = clf.tree_

  # List to store the final string specifications (paths + morpheme)
  specifications = []

  # Recursive function to traverse the tree
  def recurse(node, path):

    # If this node is not a leaf, keep splitting
    if tree_.feature[node] != _tree.TREE_UNDEFINED:
      feature = feature_names[tree_.feature[node]]

      # Left child (<=0.5) represented as -feature
      recurse(tree_.children_left[node], path + [f"-{feature}"])

      # Right child (> 0.5) represented as +feature
      recurse(tree_.children_right[node], path + [f"+{feature}"])

    # If it is a leaf node, append the corresponding specification to 
    # the specifications list. 
    else:
      predicted_morph = clf.classes_[np.argmax(tree_.value[node])] # identify the most common morph at leaf 
      feature_string = ", ".join(path) # feature list --> comma-separated string
      specifications.append("["+feature_string+"] <--> "+predicted_morph)
  
  #Start recursion from the root node with an empty path 
  recurse(0, [])

  #Return the list of specifications for all leaves 
  return specifications
