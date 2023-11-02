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
    eval_cross_validation(clean_data, seed)
    eval_cross_validation(noisy_data, seed)

    # Nested 10 fold cross validation of pruned tree
    pruning_tests(clean_data, seed)
    pruning_tests(noisy_data, seed)
    return None

def eval_cross_validation(data, seed): 
    """10 fold cross validation

    Args:
        data (np.ndarray): data set to be used (will be split into training & test data)
        seed (int): random number generator seed used for reproducible results

    Returns:
        None
    """
    evaluation_vector = np.zeros(10)
    confusion_matrix = np.zeros((4,4))
    for i in range(10):
        #Split the dataset
        train_data, test_data = evaluate.split_dataset(data, 0.1, seed)

        #First cross-validation evaluation bit
        decision_tree, depth = decision_tree_learning(train_data, 0)
        evaluation_vector[i] = evaluate.evaluate(test_data, decision_tree)
        confusion_matrix += evaluate.confusion_matrix(test_data[:,:-1], test_data[:,-1], decision_tree)
    
    #For the evaluation section
    average_eval = np.average(evaluation_vector)
    average_confusion_matrix = confusion_matrix / 10
    print(average_confusion_matrix, average_eval)
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
    evaluation_vector = np.zeros(10)
    confusion_matrix = np.zeros((4,4))
    
    for i in range(10):
        accuracy = -np.inf
        #Split the dataset
        train_data, test_data = evaluate.split_dataset(data, 0.1, seed)
        
        for j in range(9):
            #Pruning cross-validation bit
            cross_train_data, validation_data = evaluate.split_dataset(train_data, 1/9, seed)
            prune = pr.Prune(cross_train_data, validation_data)
            initial_tree, pruned_tree = prune.get_optimum_pruned_tree()
            if evaluate.evaluate(validation_data,pruned_tree) >= accuracy:
                accuracy = evaluate.evaluate(validation_data,pruned_tree)
                output_tree = pruned_tree
            else:
                pass
        
        evaluation_vector[i] = evaluate.evaluate(test_data, output_tree)
        confusion_matrix += evaluate.confusion_matrix(test_data[:,:-1], test_data[:,-1], output_tree)
    
    #For the evaluation section
    average_eval = np.average(evaluation_vector)
    average_confusion_matrix = confusion_matrix / 10
    print(average_confusion_matrix, average_eval)
    evaluate.print_metrics(average_confusion_matrix)
        
    return None

if __name__ == '__main__':
    seed = default_rng(444233429)
    main(seed)



