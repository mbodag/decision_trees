import numpy as np
import json

def split_dataset_with_validation(dataset, test_ratio, validation_ratio = 0.0):
    """Create a decision tree of given depth from the input dataset

    Extended description of function.

    Args:
        dataset (numpy.ndarray): The dataset to be split
        test_ratio (float): The ratio of the dataset to be used for testing
        validation_ratio (float, default 0.0): The ratio of the dataset to be used for validation

    Returns:
        training_data (numpy.ndarray): Training dataset
        test_data (numpy.ndarray): Test dataset
        validation_data (numpy.ndarray): Validation dataset (empty if validation_ratio is not specified)

    """
    if test_ratio > 1 or validation_ratio > 1 or test_ratio + validation_ratio > 1:
        raise ValueError('Invalid ratios')
    np.random.shuffle(dataset)
    num_data_points = np.size(dataset, axis = 0)
    num_test_points = round(num_data_points * test_ratio)
    test_data = dataset[:num_test_points]
    training_data = dataset[num_test_points:]
    validation_data = np.array([])
    if validation_ratio > 0:
        num_validation_points = round(num_data_points * validation_ratio)
        validation_data = training_data[:num_validation_points]
        training_data = training_data[num_validation_points:]

    
    return training_data, test_data, validation_data

def split_dataset(dataset, test_ratio):
    """Create a decision tree of given depth from the input dataset

    Extended description of function.

    Args:
        dataset (numpy.ndarray): The dataset to be split
        test_ratio (float): The ratio of the dataset to be used for testing
        validation_ratio (float, default 0.0): The ratio of the dataset to be used for validation

    Returns:
        training_data (numpy.ndarray): Training dataset
        test_data (numpy.ndarray): Test dataset
        validation_data (numpy.ndarray): Validation dataset (empty if validation_ratio is not specified)

    """
    if test_ratio > 1:
        raise ValueError('Invalid ratio')
    np.random.shuffle(dataset)
    num_data_points = np.size(dataset, axis = 0)
    num_test_points = round(num_data_points * test_ratio)
    test_data = dataset[:num_test_points]
    training_data = dataset[num_test_points:]
    
    return training_data, test_data

def predict_instance(x_instance, decision_tree):
    """Predicts the output of a given data instance using a decision tree
    
    By recursively calling the function, we traverse the decision tree using the data instance's features until a leaf node is reached, 
    whose value can be used as prediction

    Args:
        x_instance (numpy.ndarray): The features of an data instance
        decision_tree (dict): Decision tree containing rules on how to predict data points from their features. 
        Expected shape of each node: {attribute: (str) or None, value: (float), left: (dict) or None, right: (dict) or None} 
        (None values if leaf node)
                            

    Returns:
        value (int): Predicted output
    """
    if decision_tree['attribute'] is None:
        return decision_tree['value']
    else:
        attribute = int(decision_tree['attribute'][1])
        value = decision_tree['value']
        if x_instance[attribute] < value:
                return predict_instance(x_instance, decision_tree['left'])
        elif x_instance[attribute] >= value: #NOTE IF the instance value is equal to the decision tree value then we take the node on the right arbitrarily
                return predict_instance(x_instance, decision_tree['right'])
        else:
            print('This issue is unaccounted for')
        
def predict(x_test, decision_tree):
    """Predicts the output of a given feature dataset using a decision tree
    
    Predicts individual instances and adds them to a numpy ndarray

    Args:
        x_test (numpy.ndarray): A feature dataset of size n x m where n is the number of instances m is the number of features
        decision_tree (dict): Decision tree containing rules on how to predict data points from their features. 
        Expected shape of each node: {attribute: (str) or None, value: (float), left: (dict) or None, right: (dict) or None} 
        (None values if leaf node)
                            
    Returns:
        y_predicted (int): An 1-d array of size n where the value at index i corresponds to the predicted category of instance i from x_test
    """
    num_data_points = np.size(x_test, axis = 0)
    y_predicted = np.empty(num_data_points)
    for instance in range(num_data_points):
        y_predicted[instance] = predict_instance(x_test[instance], decision_tree)
    return y_predicted

def confusion_matrix(x_test, y_test, decision_tree):
    """Creates a confusion matrix from the data"""

    y_predicted = predict(x_test, decision_tree)
    confusion_matrix = np.zeros((4,4)) #TODO CHANGE LATER TO GENERALISE
    for i in range(np.size(y_predicted)):
        try:
            confusion_matrix[int(y_test[i]) - 1,int(y_predicted[i]) - 1] = confusion_matrix[int(y_test[i])-1,int(y_predicted[i])-1] + 1
        except:
            print(confusion_matrix[int(y_test[i]) - 1,int(y_predicted[i]) - 1])
            continue
    return confusion_matrix

def accuracy(confusion_matrix):
    """Returns accuracy from a confusion matrix"""
    return np.trace(confusion_matrix) / np.sum(confusion_matrix)

def precision(confusion_matrix):
    """Returns precision from a confusion matrix"""
    return np.diag(confusion_matrix) / np.sum(confusion_matrix, axis = 1)

def recall(confusion_matrix):
    """Returns recall from a confusion matrix"""
    return np.diag(confusion_matrix) / np.sum(confusion_matrix, axis = 0)

def macro_avg_precision(confusion_matrix):
    """Returns macro averaged precision from a confusion matrix"""
    return np.average(np.diag(confusion_matrix) / np.sum(confusion_matrix, axis = 1))

def macro_avg_recall(confusion_matrix):
    """Returns macro averaged recall from a confusion matrix"""
    return np.average(np.diag(confusion_matrix) / np.sum(confusion_matrix, axis = 0))

def print_metrics(confusion_matrix):
    """Prints important metrics from a confusion matrix"""
    print("Accuracy: "+ str(accuracy(confusion_matrix)))
    print("Precision: "+ str(precision(confusion_matrix)))
    print("Recall: "+ str(recall(confusion_matrix)))
    print("Macro averaged precision: "+ str(macro_avg_precision(confusion_matrix)))
    print("Macro averaged recall: "+ str(macro_avg_recall(confusion_matrix)))

#TODO write this function
def f_score(confusion_matrix, beta):
    pass

def evaluate(test_db, trained_tree):
    pass

    
