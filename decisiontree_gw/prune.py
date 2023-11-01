import numpy as np
import decisiontree_gw.train as tr
import copy
import evaluate

class Prune:
    def __init__(self, training_data, validation_data):
        """Class to prune decision trees
        """
        dt = tr.DecisionTreeTrain()
        self.tree, self.tree_depth = dt.decision_tree_learning(training_data)
        self.training_data = training_data
        self.validation_data = validation_data

    def get_optimum_pruned_tree(self):
        """Method to prune and get best validation error performance with the set of training and validation data

        Returns:
            pruned_tree (dict):
        """
        initial_tree = copy.deepcopy(self.tree)
        best_accuracy = -np.inf
        tree = self.tree
        classified_nodes, list_to_prune = self.get_list_to_prune(tree)
        checked = set() # list of nodes that are checked but not pruned

        while list_to_prune:
            if list(list_to_prune[0].values())[0] not in checked:
                pruned_tree = self.prune_next(tree, classified_nodes, list_to_prune[0])
                if best_accuracy <= evaluate.evaluate(self.validation_data, pruned_tree): # to be replaced with validation error logic -> chat with Matis
                    best_accuracy = evaluate.evaluate(self.validation_data, pruned_tree)
                    tree = pruned_tree
                else:
                    checked.add(list(list_to_prune[0].values())[0])

            classified_nodes, list_to_prune_update = self.get_list_to_prune(tree)
            list_to_prune = []
            for node in list_to_prune_update:
                if list(node.values())[0] not in checked:
                    list_to_prune.append(node)
                else:
                    pass
        
        return (initial_tree,tree)

    def prune_next(self, initial_tree, classified_nodes, node_to_prune):
        """Method to prune and return updated tree:
        
        Arg:
            tree (dict): original or pruned tree of self.tree
            node_to_prune (dict): dictionary representing the node to prune
            classified_nodes (dict): nodes indexed by id

        Returns:
            pruned_tree (dict):
        """
        tree = copy.deepcopy(initial_tree)
        # Get majority value of two leaf
        majority_len = max([node_to_prune["tree"]["left"]["len"], node_to_prune["tree"]["right"]["len"]])
        for leaf in [node_to_prune["tree"]["left"], node_to_prune["tree"]["right"]]:
            if leaf["len"] == majority_len:
                majority_value = leaf["value"]
            else:
                pass

        id_path=[node_to_prune["id"]]
        parent_id = node_to_prune["parent_id"]
        # Trace back parents of nodes step by step back to top of tree
        while parent_id != '0':
            id_path.insert(0,parent_id)
            parent_id = classified_nodes[parent_id]["parent_id"]
        
        lr_path = []
        left = tree["left"]
        right = tree["right"]
        # Construct left - right path from top of tree to location of node to prune
        for item in id_path:
            if left == classified_nodes[item]["tree"]:
                lr_path.append("left")
                temp = left
                left = temp["left"]
                right = temp["right"]
            elif right == classified_nodes[item]["tree"]:
                lr_path.append("right")
                temp = right
                left = temp["left"]
                right = temp["right"]
            else:
                lr_path.append(None)
        # Update node that need to be pruned
        node = tree
        #try:
        for i in range(len(lr_path)):
            if i == len(lr_path)-1:
                node[lr_path[i]]["attribute"] = None
                node[lr_path[i]]["left"] = None
                node[lr_path[i]]["right"] = None
                node[lr_path[i]]["value"] = majority_value
            else:
                node = node[lr_path[i]]      
        #except:
            #print("Node not found")   
        return tree
    
    def compare_dicts(self, dict1, dict2):
        """Simple method to compare two dictionaries (decision trees) have the same overlapping keys

        Args:
            dict1(dict):
            dict2(dict):

        Returns:
            True / False
        """
        # Get the keys that are common to both dictionaries
        common_keys = set(dict1.keys()).intersection(dict2.keys())

        # Check if all the values for common keys are the same
        for key in common_keys:
            if dict1[key] != dict2[key]:
                return False  # Values for a common key are different
    
        return True  # All overlapping keys have the same values

    def get_list_to_prune(self, tree_initial):
        """Method to identify node to prune based on criterias:
            1. Node directly connects to two leaves
            2. Node is at deepest level of tree

        Returns:
            classified_nodes (dict): dictionary of nodes with properties indexed by id
            list_to_prune (dict): the node to be pruned sorted by priority
        """
        tree = copy.deepcopy(tree_initial)
        classified_nodes = self.classify_nodes(tree)
        all_leaf_nodes = []

        for id, node in classified_nodes.items():
            if node["type"] == "leaf_node":
                all_leaf_nodes.append(node)

        list_to_prune = sorted(all_leaf_nodes, key=lambda k: k["tree"]["depth"], reverse=True)
        return (classified_nodes, list_to_prune)

    def classify_nodes(self, tree_initial):
        """Method to classify_nodes, giving it an ID, identify leaf nodes and depth of nodes

        Returns:
            classified_nodes (dict): dictionary index by an id and sub dictionary of decision tree below the node
        """
        tree = copy.deepcopy(tree_initial)
        id = 0
        tree["id"] = id
        tree["parent_id"] = None
        nodes = [{"0":tree}]
        classified_nodes = {}
        # First item in classified_nodes should be the entire tree initialised below
        properties_0 = {}
        properties_0["id"] = id
        properties_0["tree"] = tree
        properties_0["type"] = "origin"
        classified_nodes[str(id)] = properties_0

        # Iterate through items in nodes to classify them until empty   
        while nodes != []:
            add_nodes, properties = self.get_sub_tree_properties(list(nodes[0].values())[0],id)
            for property in properties:
                if property == None:
                    pass
                else:
                    property
                    property["parent_id"] = list(nodes[0].keys())[0]
                    classified_nodes[str(property["id"])] = property
                    id += 1
            nodes.remove(nodes[0])
            for item in add_nodes:
                if list(item.values())[0] == None:
                    pass
                else:
                    nodes.append(item)
        return classified_nodes
        
    def get_sub_tree_properties(self, tree_initial, id):
        """Method to get the properties of sub trees (brances) of a tree

        Args:
            tree (dict): dictionary representing part of decision tree
            id: id for highest level node in the input tree

        Returns:
            branches_out (list): left branch (dict) and right branch (dict) of a tree indexed by ID
            properties (dict): list of properties (dict) for left and right branch indexed by ID
        """
        tree = copy.deepcopy(tree_initial)
        left_key = "left"
        right_key = "right"
        sub_tree_left = tree[left_key]
        sub_tree_right = tree[right_key]
        branches = [sub_tree_left, sub_tree_right]
        branches_out = []
        properties = []
        for branch in branches:
            # When tree == None, this branch is already a leaf, not a node anymore
            if branch != None:
                node_properties={}
                id += 1
                node_properties["id"] = str(id)
                node_properties["tree"] = branch
                # If both branch are None, this node is a leaf
                if branch[left_key] == None and branch[right_key] == None:
                    node_properties["type"] = "leaf"
                # If node is connected to 2 leaves, it's a leaf_node
                elif (branch[left_key][left_key] == None and
                      branch[left_key][right_key] == None and
                      branch[right_key][left_key] == None and
                      branch[right_key][right_key] == None):
                    node_properties["type"] = "leaf_node"
                # Otherwise its a normal branch
                else:
                    node_properties["type"] = "branch_node"

                properties.append(node_properties)
            else:
                properties.append(None)
            branches_out.append({str(id):branch})
        
        return (branches_out, properties)
        
def test_prune_optimum():
    data = np.loadtxt("wifi_db/clean_dataset.txt")
    pr = Prune(data,data)
    it, pt = pr.get_optimum_pruned_tree()
    print(it)
    print(pt)

if __name__ == "__main__":
    test_prune_optimum()