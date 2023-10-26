from algorithm_matis import decision_tree_learning
import evaluate
import numpy as np
import json

#TODO Clean up this file
def main():
    #Load data
    clean_data = np.loadtxt('wifi_db/clean_dataset.txt')
    noisy_data = np.loadtxt('wifi_db/noisy_dataset.txt')

    #Split the dataset
    evaluation_vector = np.zeros(10)
    for i in range(10):
        train_data, test_data = evaluate.split_dataset(clean_data, 0.1)

        #First cross-validation evaluation bit
        decision_tree, depth = decision_tree_learning(train_data, 0)
        evaluation_vector[i] = evaluate.evaluate(test_data, decision_tree)
        
        for j in range(10):
            #Pruning cross-validation bit
            cross_train_data, validation_data = evaluate.split_dataset(train_data, 0.99) #
            #I realised I split the train_data and put the training bit back into train_data which exponentially decreased the amount of training data on every loop. 
            #Took me an hour to figure out what was wrong, I thought it was my decision tree algorithm. 
            #Therefore I'm renaming the variable to cross_train_data, but feel free to change it
            decision_tree, depth = decision_tree_learning(cross_train_data, 0)
    
    #For the evaluation section
    average_first_eval = np.average(evaluation_vector)

    #file = open('example_tree.json','w')
    #json.dump(decision_tree, file, indent = 4)
    conf_matrix = evaluate.confusion_matrix(test_data[:,:-1], test_data[:,-1], decision_tree)
    print(conf_matrix)
    evaluate.print_metrics(conf_matrix)


if __name__ == '__main__':
    main()



