import numpy as np
class DecisionTreeTrain:
    def __init__(self):
        """Class to train decision tree
        """
    
    def decision_tree_learning(self,training_data,depth=0):
        """Method to create decision tree based on the training data input

        Args:
            depth (int): starting depth of decision tree (default to be zero)
        
        Returns:
            decision_tree (dict): decision tree output in dictionary
        """
        try:
            labels = np.unique(training_data[:,-1])
            if len(labels) == 1:
                return ({'attribute': None, 'value': np.floor(labels[0]), 'left': None, 'right': None, 'depth':depth, 'len': len(training_data)}, depth+1)
            else:
                split_attribute, split = self.find_split(training_data)
                left_data = training_data[training_data[:,split_attribute-1]<=split]
                right_data = training_data[training_data[:,split_attribute-1]>split]
                left_branch, left_depth = self.decision_tree_learning(left_data,depth+1)
                right_branch, right_depth = self.decision_tree_learning(right_data, depth+1)
                node = {'attribute': 'X'+str(np.floor(split_attribute)), 'value': split, 'left': left_branch, 'right': right_branch, 'depth': depth+1, 'len': len(training_data)}
            return (node, max(left_depth,right_depth))
        except:
            print("Error occured, check data input, returnning empty tree")
            return {}


    def find_split(self, data):
        """Method to find split point of attributes that gives the highest information gain

        Args:
            data (np.array): an numpy array of training dataset with each column representing a attribute and last column representing the label
        
        Returns:
            best_attribute (int): attribute resulting in highest information gain
            best_split_value (float): split value resulting in hightest information gain
        """
        
        attributes = np.shape(data)[-1]-1
        overall_entropy = self.calculate_entropy(data)
        best_information_gain = -np.Inf
        best_attribute = None
        best_split_value = None

        for attribute in range(1,attributes+1):
            potential_splits = np.sort(data[:,attribute-1])
            for split in np.unique(potential_splits):
                left = data[data[:,attribute-1]<=split]
                right = data[data[:,attribute-1]>split]
                information_gain = (
                    overall_entropy - 
                    len(left)/len(data)*self.calculate_entropy(left) - 
                    len(right)/len(data)*self.calculate_entropy(right)
                    )
                if information_gain > best_information_gain:
                    best_information_gain = information_gain
                    best_attribute = attribute
                    best_split_value = split
        return (best_attribute,best_split_value)
    
    def calculate_entropy(self, data,):
        """calculate_entropy for a given data extract

        Args:
            data (np.array): an numpy array of training dataset with each column representing a attribute and last column representing the label
        
        Returns:
            entropy (float)
        """
        labels = np.unique(data[:,-1])
        entropy = 0
        for label in labels:
            label_occurance = len(data[data[:,-1]==label])
            data_size = len(data)
            entropy += -label_occurance/data_size * np.log2(label_occurance/data_size)
        return entropy

def test_train():
    """Test the class runs
    """
    training_data = np.loadtxt("wifi_db/clean_dataset.txt")
    dt = DecisionTreeTrain()
    tree, depth = dt.decision_tree_learning(training_data)
    print(tree)
    print(depth)

if __name__ == "__main__":
    test_train()

