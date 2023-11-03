import main as dt
import numpy as np
import matplotlib.pyplot as plt

# We want to visualise the full decision tree based on the clean dataset
clean_data = np.loadtxt('wifi_db/clean_dataset.txt')
decision_tree, depth = dt.decision_tree_learning(clean_data, 0)


def plot_tree(node, x, y, parent, branch_text, depth=0, max_depth=4):
    """ Plots a decision tree up to a given depth using recursion.
    
    Arguments:
      node (dict) : Node of decision tree nested in the dictionary.
      x (int) : x co-ordinate of where to begin the plot in the x-y plane
      y (int) : y co-ordinate of where to begin the plot in the x-y plane
      parent (tuple) : x-y co-orindates of parent node
      branch_text (str) : node label, should use in-line mathematical syntax
      depth (int) : counter to keep track of how deep the tree is so far
      max_depth (int) : a termination condition for maximum tree depth
      
    Outputs:
      (plt.annotate) : a series of annotations for initialised matplotlib fig
    
    """
    if depth > max_depth:
        return
    
    if node['attribute'] is not None:
        text = '$' + node['attribute'][0] + '_' + node['attribute'][1] + ' < ' + str(node['value']) + '$'
    else:
        text = '$X_' + str(node['value']) + '$'
        plt.scatter(x, y, marker='o', color='green', s=200, alpha = 0.5)
    
    # Adjust the spacing between the arrows
    spacing = 2
    
    # Plot the node
    if node['attribute'] is not None:
            plt.annotate(text, 
                          xy=(parent[0], 
                              parent[1]-0.1), 
                          xytext=(x, y),
                          arrowprops=dict(arrowstyle="<-", 
                                          connectionstyle="arc3", 
                                          color='black', 
                                          shrinkA=5, 
                                          shrinkB=5),
                          bbox=dict(boxstyle="round,pad=0.3",
                                    fc="white",
                                    ec="black",
                                    lw=1),
                          horizontalalignment='center', 
                          verticalalignment='center', 
                          fontsize=8)
    else:
        plt.annotate(text, 
                      xy=(parent[0],
                          parent[1]-0.1),
                      xytext=(x, y),
                      arrowprops=dict(arrowstyle="<-",
                                      connectionstyle="arc3",
                                      color='black',
                                      shrinkA=5,
                                      shrinkB=5),
                      horizontalalignment='center',
                      verticalalignment='center',
                      fontsize=8)

    
    # Plot left and right branches
    if node['left'] is not None:
        plot_tree(node['left'],
                  x - spacing * 2 ** (max_depth - depth - 1),
                  y - 1,
                  (x, y),
                  'L',
                  depth + 1)
    if node['right'] is not None:
        plot_tree(node['right'],
                  x + spacing * 2 ** (max_depth - depth - 1),
                  y - 1,
                  (x, y),
                  'R',
                  depth + 1)



# Now plot the full tree
def plot_full_tree(node, x, y, parent, branch_text, depth=0, max_depth=16):
    """ Plots the full decision tree (i.e. up to 16 layers depth).
    
    This is very similar to the function above other than that some of the
    constants are adjusted to try to best fit on the nodes in the neatest
    way possible. Using a separate function allows us to keep these distinct.
    
    Arguments:
      node (dict) : Node of decision tree nested in the dictionary.
      x (int) : x co-ordinate of where to begin the plot in the x-y plane
      y (int) : y co-ordinate of where to begin the plot in the x-y plane
      parent (tuple) : x-y co-orindates of parent node
      branch_text (str) : node label, should use in-line mathematical syntax
      depth (int) : counter to keep track of how deep the tree is so far
      max_depth (int) : a termination condition for maximum tree depth
      
    Outputs:
      (plt.annotate) : a series of annotations for initialised matplotlib fig
    
    """
    if depth > max_depth:
        return
    
    if node['attribute'] is not None:
        text = '$' + node['attribute'][0] + '_' + node['attribute'][1] + ' < ' + str(node['value']) + '$'
    else:
        text = '$X_' + str(node['value']) + '$'
        plt.scatter(x, y, marker='o', color='green', s=200, alpha = 0.5)
    
    # Adjust the spacing between the arrows
    spacing = 2
    
    # Plot the node
    if node['attribute'] is not None:
            plt.annotate(text, 
                         xy=(parent[0], 
                             parent[1]-0.1), 
                         xytext=(x, y),
                         arrowprops=dict(arrowstyle="<-", 
                                         connectionstyle="arc3", 
                                         color='black', 
                                         shrinkA=5, 
                                         shrinkB=5),
                         bbox=dict(boxstyle="round,pad=0.3",
                                   fc="white",
                                   ec="black",
                                   lw=1),
                         horizontalalignment='center', 
                         verticalalignment='center', 
                         fontsize=8)
    else:
        plt.annotate(text, 
                     xy=(parent[0],
                         parent[1]-0.1),
                     xytext=(x, y),
                     arrowprops=dict(arrowstyle="<-",
                                     connectionstyle="arc3",
                                     color='black',
                                     shrinkA=5,
                                     shrinkB=5),
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=8)

    
    # Plot left and right branches
    if node['left'] is not None:
        plot_full_tree(node['left'],
                  x - spacing * 2 ** (max_depth - depth - 1),
                  y - 1,
                  (x, y),
                  'L',
                  depth + 1)
    if node['right'] is not None:
        plot_full_tree(node['right'],
                  x + spacing * 2 ** (max_depth - depth - 1),
                  y - 1,
                  (x, y),
                  'R',
                  depth + 1)

if __name__ == '__main__':
    # Initialize the plot
    # Adjust the figure size and use subplots
    fig = plt.figure(figsize=(12, 6))

    #Starting point for the root node
    plot_tree(decision_tree, 0, 0, (0, 0), 'Root', 1)

    #Adjusting plot limits
    plt.ylim(-4, 0)
    plt.xlim(-12, 15)
    plt.axis('off')

    fig.savefig('decision_tree_graph.svg', format = 'svg', dpi = 600)

    fig = plt.figure(figsize=(12, 6))

    # Call the plot_tree function
    plot_full_tree(decision_tree, 0, 0, (0, 0), 'Root', 1)

    plt.ylim(-15, 1)
    plt.xlim(-65000,65000)
    plt.axis('off')

    fig.savefig('decision_tree_full_graph.svg', format = 'svg', dpi = 600)
