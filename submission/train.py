import numpy as np


def calculate_entropy(dataset):
    """Calculate information entropy

    Args:
        dataset (np.ndarray): data set to be used as training data

    Returns:
        entropy (float): information entropy
    """
    values, counts = np.unique(dataset[:,-1], return_counts=True)
    probabilities = counts / np.sum(counts)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def find_split(dataset):
    """Finding the split of left and right branches based on information gain

    Args:
        dataset (np.ndarray): data set to be used as training data

    Returns:
        optimal attribute (float): ID of best atribute to split the data
        optimal_value (float): best value of optimal attribute to split branches
    """
    min_entropy = np.inf
    optimal_value = None
    optimal_attribute = None
    num_data_points = np.size(dataset, axis = 0)

    for attribute in range(np.size(dataset, axis = 1) - 1):
        dataset = dataset[dataset[:, attribute].argsort()]
        for cut in range(1,num_data_points):
            if dataset[cut-1,attribute] == dataset[cut, attribute]:
                    continue
            left_dataset = dataset[:cut]
            right_dataset = dataset[cut:]
            num_left_data_points = np.size(left_dataset, axis = 0)
            num_right_data_points = np.size(right_dataset, axis = 0)

            sum_entropy = (num_left_data_points/num_data_points) * calculate_entropy(left_dataset) + (num_right_data_points/num_data_points)* calculate_entropy(right_dataset)
            if sum_entropy < min_entropy:
                min_entropy = sum_entropy
                optimal_value = (dataset[cut-1,attribute] + dataset[cut, attribute])/2 
                optimal_attribute = attribute
    #print(dataset, optimal_value, optimal_attribute)
    return optimal_attribute, optimal_value



def decision_tree_learning(dataset: np.array, depth: int):
    """Create a decision tree of given depth from the input dataset

    Extended description of function.

    Args:
        dataset (numpy.ndarray): The dataset to create the tree from
        depth (int): The depth of the decision tree

    Returns:
        node (dict): Description of return value
        depth (int)

    """
    if len(np.unique(dataset[:,-1])) == 1:
        return({'attribute' : None ,
                'value' : int(dataset[0,-1]), 
                'left': None, 
                'right': None,
                'depth': depth,'len':len(dataset)}, depth) #Edge case?
    else:
        attribute_index, value = find_split(dataset)
        left_dataset = dataset[dataset[:,attribute_index] < value]
        right_dataset = dataset[dataset[:, attribute_index] > value]
        #print(len(left_dataset)+len(right_dataset) == len(dataset), len(left_dataset), len(right_dataset)) #Might need a >=
        left_branch, left_depth = decision_tree_learning(left_dataset, depth+1)
        right_branch, right_depth = decision_tree_learning(right_dataset, depth+1)
        node = {'attribute': 'X'+str(attribute_index), 'value': value, 'left': left_branch, 'right': right_branch, 'depth':depth, 'len':len(dataset)}
        return(node, max(left_depth, right_depth))