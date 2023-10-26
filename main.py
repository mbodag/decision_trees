from algorithm_matis import decision_tree_learning 
import evaluate
import numpy as np
import json

#TODO Clean up this file

if __name__ == '__main__':
    clean_data = np.loadtxt('wifi_db/clean_dataset.txt')
    #noisy_data = np.loadtxt('wifi_db/noisy_dataset.txt')
    train_data, test_data = evaluate.split_dataset(clean_data, 0.8)
    decision_tree, depth = decision_tree_learning(train_data, 0)
    file = open('example_tree.json','w')
    json.dump(decision_tree, file, indent = 4)
    conf_matrix = evaluate.confusion_matrix(test_data[:,:-1], test_data[:,-1], decision_tree)
    print(conf_matrix)
    evaluate.print_metrics(conf_matrix)




