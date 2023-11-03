from train import decision_tree_learning
import evaluate
import numpy as np
import json
from numpy.random import default_rng
import prune as pr

def main(seed):
    """Running cross validation and output performance metrics

    Args:
        seed (int): random number generator seed used for reproducible results

    Returns:
        None
    """
    clean_data = np.loadtxt('wifi_db/clean_dataset.txt')
    noisy_data = np.loadtxt('wifi_db/noisy_dataset.txt')

    # 10 fold cross validation of unpruned tree on clean and noisy data
    print("Output for decision tree training algorithm without pruning - Clean data")
    eval_cross_validation(clean_data, seed)
    print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-")
    print("Output for decision tree training algorithm without pruning - Noisy data")
    eval_cross_validation(noisy_data, seed)
    print("=========================================================================")

    # Nested 10 fold cross validation of pruned tree
    print("Output for decision tree training algorithm with pruning - Clean data")
    pruning_tests(clean_data, seed)
    print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-")
    print("Output for decision tree training algorithm with pruning - Noisy data")
    pruning_tests(noisy_data, seed)
    print("=========================================================================")
    return None

def eval_cross_validation(data, seed): 
    """10 fold cross validation

    Args:
        data (np.ndarray): data set to be used (will be split into training & test data)
        seed (int): random number generator seed used for reproducible results

    Returns:
        None
    """
    confusion_matrix = np.zeros((4,4))
    for i in range(10):
        #Split the dataset
        train_data, test_data = evaluate.split_dataset(data, 0.1, seed)

        #First cross-validation evaluation bit
        decision_tree, depth = decision_tree_learning(train_data, 0)
        confusion_matrix += evaluate.confusion_matrix(test_data[:,:-1], test_data[:,-1], decision_tree)
    
    #For the evaluation section
    average_confusion_matrix = confusion_matrix / 10
    print("Average confusion matrix: \n", average_confusion_matrix)
    evaluate.print_metrics(average_confusion_matrix)
    #file = open('example_tree.json','w')
    #json.dump(decision_tree, file, indent = 4)
    return None


def pruning_tests(data, seed):
    """Nested 10 fold cross validation

    Args:
        data (np.ndarray): data set to be used (will be split into training & test data)
        seed (int): random number generator seed used for reproducible results

    Returns:
        None
    """
    confusion_matrix = np.zeros((4,4))
    all_unpruned_depth = []
    all_pruned_depth = []
    
    for i in range(10):
        accuracy = -np.inf
        #Split the dataset
        train_data, test_data = evaluate.split_dataset(data, 0.1, seed)
        each_pruned_depth = []
        each_unpruned_depth = []
        for j in range(9):
            #Pruning cross-validation bit
            cross_train_data, validation_data = evaluate.split_dataset(train_data, 1/9, seed)
            prune = pr.Prune(cross_train_data, validation_data)
            initial_tree,initial_depth, pruned_tree, pruned_depth = prune.get_optimum_pruned_tree()
            each_unpruned_depth.append(initial_depth)
            each_pruned_depth.append(pruned_depth)
            # Calculate average confution matrix and other performance metrics over 90 trees (nested 10 fold cross validation)
            confusion_matrix += evaluate.confusion_matrix(test_data[:,:-1], test_data[:,-1], pruned_tree)
        all_pruned_depth.append(np.average(each_pruned_depth))
        all_unpruned_depth.append(np.average(each_unpruned_depth))
        
    
    #For the evaluation section
    average_confusion_matrix = confusion_matrix / 90
    print("Average confusion matrix: \n", average_confusion_matrix)
    evaluate.print_metrics(average_confusion_matrix)
    print("average pre-pruning depth", np.average(all_unpruned_depth))
    print("average post-pruning depth", np.average(all_pruned_depth))
        
    return None

if __name__ == '__main__':
    #seed used in report: 444233429
    seed = default_rng(444233429)
    main(seed)