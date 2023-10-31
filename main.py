from algorithm_matis import decision_tree_learning
import evaluate
import numpy as np
import json
from numpy.random import default_rng
#TODO Clean up this file
def main(seed):
    
    #Load data
    clean_data = np.loadtxt('wifi_db/clean_dataset.txt')
    noisy_data = np.loadtxt('wifi_db/noisy_dataset.txt')

    eval_cross_validation(clean_data, seed)
    eval_cross_validation(noisy_data, seed)

def eval_cross_validation(data, seed): 
    
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
    #file = open('example_tree.json','w')
    #json.dump(decision_tree, file, indent = 4)


def pruning_tests(data, seed):
    for i in range(10):
        #Split the dataset
        train_data, test_data = evaluate.split_dataset(data, 0.1, seed)
        
        for j in range(10):
            #Pruning cross-validation bit
            cross_train_data, validation_data = evaluate.split_dataset(train_data, 0.9, seed)
            decision_tree, depth = decision_tree_learning(cross_train_data, 0)

if __name__ == '__main__':
    seed = default_rng(444233429)
    main(seed)



