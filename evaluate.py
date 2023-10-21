import numpy as np
import json

def split_dataset(dataset, split_ratio):
    np.random.shuffle(dataset)
    num_data_points = np.size(dataset, axis = 0)
    num_training_points = round(num_data_points * split_ratio)
    training_data = dataset[:num_training_points]
    test_data = dataset[num_training_points:]
    return training_data, test_data

def predict_instance(x_instance, trained_tree):
    if trained_tree['attribute'] is None:
        return trained_tree['value']
    else:
        attribute = int(trained_tree['attribute'][1])
        value = trained_tree['value']
        if x_instance[attribute] < value:
                return predict_instance(x_instance, trained_tree['left'])
        elif x_instance[attribute] >= value: #NOTE IF the instance value is equal to the decision tree value then we take the node on the right arbitrarily
                return predict_instance(x_instance, trained_tree['right'])
        else:
            print('This issue is unaccounted for')
        
def predict(x_test, trained_tree):
    num_data_points = np.size(x_test, axis = 0)
    y_predicted = np.empty(num_data_points)
    for instance in range(num_data_points):
        y_predicted[instance] = predict_instance(x_test[instance], trained_tree)
    return y_predicted

def confusion_matrix(x_test, y_test, trained_tree):
    y_predicted = predict(x_test, trained_tree)
    confusion_matrix = np.zeros((4,4)) #TODO CHANGE LATER TO GENERALISE
    for i in range(np.size(y_predicted)):
        try:
            confusion_matrix[int(y_test[i]) - 1,int(y_predicted[i]) - 1] = confusion_matrix[int(y_test[i])-1,int(y_predicted[i])-1] + 1
        except:
            print(confusion_matrix[int(y_test[i]) - 1,int(y_predicted[i]) - 1])
            continue
    return confusion_matrix

def accuracy(confusion_matrix):
    return np.trace(confusion_matrix) / np.sum(confusion_matrix)

def precision(confusion_matrix):
    return np.diag(confusion_matrix) / np.sum(confusion_matrix, axis = 1)

def recall(confusion_matrix):
    return np.diag(confusion_matrix) / np.sum(confusion_matrix, axis = 0)

def macro_avg_precision(confusion_matrix):
    return np.average(np.diag(confusion_matrix) / np.sum(confusion_matrix, axis = 1))

def macro_avg_recall(confusion_matrix):
    return np.average(np.diag(confusion_matrix) / np.sum(confusion_matrix, axis = 0))

def print_metrics(confusion_matrix):
    print("Accuracy: "+ str(accuracy(confusion_matrix)))
    print("Precision: "+ str(precision(confusion_matrix)))
    print("Recall: "+ str(recall(confusion_matrix)))
    print("Macro averaged precision: "+ str(macro_avg_precision(confusion_matrix)))
    print("Macro averaged recall: "+ str(macro_avg_recall(confusion_matrix)))


if __name__ == '__main__':
    pass

    
