The folder comtains all functions required and a main.py to run all coursework requirements.

To run the 10 fold cross validated assessment of decision tree algorithm performance (before and after pruning) using both clean and noisy data, simply run the main.py file.
Output of performance metrices will be displayed in terminal output.

wifi_db contains the data
train.py file contains functions used to train the decision tree.
prune.py file contains functions to prune the trained decision tree until the optimal accuracy / validation error is obtained.
evaluate.py file contains functions to run prediction on test datasets as well as functions to calculate all evaluation metrices.
visualise.py file contains functions to plot the decision tree. When it is run, figures of the tree are saved as svg files.
output.txt contains the terminal output from main.py (in case the code takes too long to run)