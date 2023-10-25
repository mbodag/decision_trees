import numpy as np


def calculate_entropy(dataset):
    value, counts = np.unique(dataset[:,-1], return_counts = True)
    label_fraction = counts/np.sum(counts)
    entropy = -np.sum(label_fraction * np.log2(label_fraction))
    return entropy

    
def get_information_gain(full_dataset, left_dataset, right_dataset):
    H_D = calculate_entropy(full_dataset)
    n = np.shape(full_dataset)[0]
    H_L = calculate_entropy(left_dataset)
    n_L = np.shape(left_dataset)[0]
    H_R = calculate_entropy(right_dataset)
    n_R = np.shape(right_dataset)[0]
    remainder_LR = (n_L/n)*H_L + (n_R/n)*H_R
    IG = H_D - remainder_LR
    return IG


def is_leaf(dataset):
    return len(np.unique(dataset[:,-1])) == 1
    

def find_split(training_data):
    n_features = np.shape(training_data)[1]-1
    splits = list()
    
    for f in range(n_features):
        sorted_data = training_data[training_data[:, f].argsort()]
        n_rows = np.shape(sorted_data)[0]
        IG_max = 0
        split_val = None
        feature = int(f)
        
        for i in range(n_rows-1):
            if sorted_data[i, f] == sorted_data[i+1, f]:
                continue
            left_dataset = sorted_data[0:i + 1]
            right_dataset = sorted_data[i+1:]
            IG = get_information_gain(sorted_data, left_dataset, right_dataset)
            if IG > IG_max:
                IG_max = IG
                split_val = (left_dataset[-1, f] + right_dataset[0, f])/2
        
        splits.append({'attribute': "X" + str(feature),
                       'IG':IG_max,
                       'value':split_val,
                       'leaf': is_leaf(sorted_data)})
    return splits


def get_max_IG(node_list):   
    IG_max = 0
    index_to_return = 0
    for i, value in enumerate(node_list):
        if value['IG'] > IG_max:
            IG_max = value['IG']
            index_to_return = i
    return node_list[index_to_return]
   

def get_split(dataset):
    return get_max_IG(find_split(dataset))


def decision_tree_learning(dataset, depth):
    if is_leaf(dataset):
        return  ({'attribute': None,
                 'value': int(dataset[0, -1]),
                 'left': None,
                 'right': None},
                 depth)
    
    else:
        split = get_split(dataset)
        attribute_index = int(split['attribute'][-1])
        value = split['value']
        left_dataset = dataset[dataset[:, attribute_index] <= value]
        right_dataset = dataset[dataset[:, attribute_index] > value]
        left_branch, left_depth = decision_tree_learning(left_dataset, depth+1)
        right_branch, right_depth = decision_tree_learning(right_dataset, depth+1)
        node = {'attribute': 'X'+str(attribute_index),
                'value': value,
                'left': left_branch,
                'right': right_branch}
        return (node, max(left_depth, right_depth))

clean_data = np.loadtxt('wifi_db/clean_dataset.txt')         
decision_tree_MH, depth_MH = decision_tree_learning(clean_data, 0)


#############################################################################

def calculate_entropy(dataset):
    values, counts = np.unique(dataset[:,-1], return_counts=True)
    probabilities = counts / np.sum(counts)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def find_split(dataset):
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
    # node = {'attribute': None, 'value': None, 'left': None, 'right': None}
    if len(np.unique(dataset[:,-1])) == 1:
        return({'attribute' : None , 'value' : int(dataset[0,-1]), 'left': None, 'right': None}, depth) #Edge case?
    else:
        attribute_index, value = find_split(dataset)
        left_dataset = dataset[dataset[:,attribute_index] < value]
        right_dataset = dataset[dataset[:, attribute_index] > value] #Might need a >=
        left_branch, left_depth = decision_tree_learning(left_dataset, depth+1)
        right_branch, right_depth = decision_tree_learning(right_dataset, depth+1)
        node = {'attribute': 'X'+str(attribute_index), 'value': value, 'left': left_branch, 'right': right_branch}
        return(node, max(left_depth, right_depth))

clean_data = np.loadtxt('wifi_db/clean_dataset.txt')
decision_tree_Matis, depth_Matis = decision_tree_learning(clean_data, 0)

assert decision_tree_MH == decision_tree_Matis
assert depth_MH == depth_Matis
