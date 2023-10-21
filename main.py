import numpy as np
import json
import evaluate


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

if __name__ == '__main__':
    clean_data = np.loadtxt('wifi_db/clean_dataset.txt')
    #noisy_data = np.loadtxt('wifi_db/noisy_dataset.txt')
    train_data, test_data = evaluate.split_dataset(clean_data, 0.8)
    decision_tree, depth = decision_tree_learning(train_data, 0)
    conf_matrix = evaluate.confusion_matrix(test_data[:,:-1], test_data[:,-1], decision_tree)
    print(conf_matrix)
    evaluate.print_metrics(conf_matrix)




