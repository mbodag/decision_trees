import main as dt
import numpy as np
import matplotlib.pyplot as plt


clean_data = np.loadtxt('wifi_db/clean_dataset.txt')
decision_tree, depth = dt.decision_tree_learning(clean_data, 0)

# We need to truncate the tree so we can plot it

def create_maths_string(dict_attribute_key, dict_value_key):
    """ Make the node labels formatted in maths text """
    try:
        string = '$' + dict_attribute_key[0] + '_' + dict_attribute_key[1] + ' < ' + str(dict_value_key) + '$'
        return string
    except:
        string = '$X_' + str(dict_value_key) + '$'
    return string

root_label = create_maths_string(decision_tree['attribute'], decision_tree['value'])

l_label = create_maths_string(decision_tree['left']['attribute'], decision_tree['left']['value'])
ll_label = create_maths_string(decision_tree['left']['left']['attribute'], decision_tree['left']['left']['value'])
lr_label = create_maths_string(decision_tree['left']['right']['attribute'], decision_tree['left']['right']['value'])
lll_label = create_maths_string(decision_tree['left']['left']['left']['attribute'], decision_tree['left']['left']['left']['value'])
llr_label = create_maths_string(decision_tree['left']['left']['right']['attribute'], decision_tree['left']['left']['right']['value'])
lrl_label = create_maths_string(decision_tree['left']['right']['left']['attribute'], decision_tree['left']['right']['left']['value'])
lrr_label = create_maths_string(decision_tree['left']['right']['right']['attribute'], decision_tree['left']['right']['right']['value'])

r_label = create_maths_string(decision_tree['right']['attribute'], decision_tree['right']['value'])
rl_label = create_maths_string(decision_tree['right']['left']['attribute'], decision_tree['right']['left']['value'])
rr_label = create_maths_string(decision_tree['right']['right']['attribute'], decision_tree['right']['right']['value'])
rrr_label = create_maths_string(decision_tree['right']['right']['right']['attribute'], decision_tree['right']['right']['right']['value'])
rrl_label = create_maths_string(decision_tree['right']['right']['left']['attribute'], decision_tree['right']['right']['left']['value'])
rlr_label = create_maths_string(decision_tree['right']['left']['right']['attribute'], decision_tree['right']['left']['right']['value'])
rll_label = create_maths_string(decision_tree['right']['left']['left']['attribute'], decision_tree['right']['left']['left']['value'])


fig = plt.figure()
ax = fig.add_subplot()
ax.axis([0, 100, 0, 100])
plt.axis('off')

# Level 1
ax.text(42, 90, root_label, size = 'xx-small', bbox = {'facecolor': 'white', 'pad':3})
ax.arrow(47, 88, -24, -10)
ax.arrow(47, 88, 24, -10)

# Level 2
ax.text(19.5, 75.5, l_label, size = 'xx-small', bbox = {'facecolor': 'white', 'pad':3})
ax.text(64.5, 75.5, r_label, size = 'xx-small', bbox = {'facecolor': 'white', 'pad':3})
ax.arrow(23, 74, -12, -18)
ax.arrow(25, 74, 8, -18)
ax.arrow(69, 74, -12, -18)
ax.arrow(71, 74, 10, -18)

# Level 3
ax.text(5, 52, ll_label, size = 'xx-small', bbox = {'facecolor': 'white', 'pad':3})
ax.text(28, 52, lr_label, size = 'xx-small', bbox = {'facecolor': 'white', 'pad':3})
ax.text(52, 52, rl_label, size = 'xx-small', bbox = {'facecolor': 'white', 'pad':3})
ax.text(77, 52, rr_label, size = 'xx-small', bbox = {'facecolor': 'white', 'pad':3})
ax.arrow(8, 50, -11, -23.5)
ax.arrow(10, 50, 7, -21.5)
ax.arrow(33, 50, -3, -18.5)
ax.arrow(35, 50, 6, -17)
ax.arrow(55, 50, -6, -18.5)
ax.arrow(57, 50, 8, -18.5)
ax.arrow(80, 50, -2, -18.5)
ax.arrow(82, 50, 9, -17)

# Level 4
ax.text(-5, 29.5, lll_label, size = 'xx-small', bbox = {'facecolor': 'white', 'pad':3})
ax.text(0, 15, '.\n.\n.', size = 'x-small', bbox = {'facecolor': 'white', 'edgecolor': 'none'}, style='oblique')
ax.text(10, 29.5, llr_label, size = 'xx-small', bbox = {'facecolor': 'white', 'pad':3})
ax.text(15, 15, '.\n.\n.', size = 'x-small', bbox = {'facecolor': 'white', 'edgecolor': 'none'}, style='oblique')
ax.text(25, 29.5, lrl_label, size = 'xx-small', bbox = {'facecolor': 'white', 'pad':3})
ax.text(30, 15, '.\n.\n.', size = 'x-small', bbox = {'facecolor': 'white', 'edgecolor': 'none'}, style='oblique')
ax.text(40, 29.5, lrr_label, size = 'xx-small', bbox = {'boxstyle': 'round,pad=0.5','facecolor': 'green', 'alpha':0.5, 'pad':0.3})
ax.text(45, 29.5, rll_label, size = 'xx-small', bbox = {'facecolor': 'white', 'pad':3})
ax.text(50, 15, '.\n.\n.', size = 'x-small', bbox = {'facecolor': 'white', 'edgecolor': 'none'}, style='oblique')
ax.text(60, 29.5, rlr_label, size = 'xx-small', bbox = {'facecolor': 'white', 'pad':3})
ax.text(65, 15, '.\n.\n.', size = 'x-small', bbox = {'facecolor': 'white', 'edgecolor': 'none'}, style='oblique')
ax.text(75, 29.5, rrl_label, size = 'xx-small', bbox = {'facecolor': 'white', 'pad':3})
ax.text(80, 15, '.\n.\n.', size = 'x-small', bbox = {'facecolor': 'white', 'edgecolor': 'none'}, style='oblique')
ax.text(90, 29.5, rrr_label, size = 'xx-small', bbox = {'boxstyle': 'round,pad=0.5', 'facecolor': 'green', 'alpha':0.5, 'pad':3})


fig.savefig('decision_tree_graph.svg', format = 'svg', dpi = 1200)