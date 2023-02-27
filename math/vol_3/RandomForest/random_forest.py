"""
Random Forest Lab

Name
Section
Date
"""
import graphviz
import os
import numpy as np
from uuid import uuid4
import time
from sklearn.ensemble import RandomForestClassifier

# Problem 1
class Question:
   """Questions to use in construction and display of Decision Trees.
   Attributes:
      column (int): which column of the data this question asks
      value (int/float): value the question asks about
      features (str): name of the feature asked about
   Methods:
      match: returns boolean of if a given sample answered T/F"""
   
   def __init__(self, column, value, feature_names):
      self.column = column
      self.value = value
      self.features = feature_names[self.column]
   
   def match(self,sample):
      """Returns T/F depending on how the sample answers the question
      Parameters:
         sample ((n,), ndarray): New sample to classify
      Returns:
         (bool): How the sample compares to the question"""
      if sample[self.column] >= self.value:
         return True
      else:
         return False

   def __repr__(self):
      return "Is %s >= %s?" % (self.features, str(self.value))
    
def partition(data,question):
   """Splits the data into left (true) and right (false)
   Parameters:
      data ((m,n), ndarray): data to partition
      question (Question): question to split on
   Returns:
      left ((j,n), ndarray): Portion of the data matching the question
      right ((m-j, n), ndarray): Portion of the data NOT matching the question
   """
   left = []
   right = []

   for row in data:
      q_bool = question.match(row)
      if q_bool:
         right.append(row)
      else:
         left.append(row)
   if len(left) == 0:
      left = None
   else:
      left = np.array(left)
   if len(right) == 0:
      right = None
   else:
      right = np.array(right)
   return left, right
   
data = np.array([[1, 1.1, 1.2],
                 [2, 2.1, 2.2]])
feature_names = ['poop', 'boogers', 'underwear']
q = Question(0, 1.1, feature_names)
left, right = partition(data, q)
# print(left)
# print(right)

#Problem 2    
def gini(data):
   """Return the Gini impurity of given array of data.
   Parameters:
        data (ndarray): data to examine
   Returns:
        (float): Gini impurity of the data"""
   label_count = np.sum(data[:,-1])
   fk_0 = (len(data) - label_count) / len(data)
   fk_1 = label_count / len(data)
   return 1 - sum([fk_0**2, fk_1**2])

def info_gain(left,right,G):
   """Return the info gain of a partition of data.
   Parameters:
      left (ndarray): left split of data
      right (ndarray): right split of data
      G (float): Gini impurity of unsplit data
   Returns:
      (float): info gain of the data"""
   len_D = len(left) + len(right)
   return G - sum([len(left) * gini(left) / len_D,
                   len(right) * gini(right) / len_D])
   
animals = np.loadtxt('animals.csv', delimiter=',')
features = np.loadtxt('animal_features.csv', delimiter=',', dtype=str,
                        comments=None)
names = np.loadtxt('animal_names.csv', delimiter=',', dtype=str)

# print(gini(animals))
# print(info_gain(animals[:50], animals[50:], gini(animals)))
# Problem 3, Problem 7
def find_best_split(data, feature_names, min_samples_leaf=5, random_subset=False):
   """Find the optimal split
   Parameters:
      data (ndarray): Data in question
      feature_names (list of strings): Labels for each column of data
      min_samples_leaf (int): minimum number of samples per leaf
      random_subset (bool): for Problem 7
   Returns:
      (float): Best info gain
      (Question): Best question"""

   max_info = 0
   best_col, best_val = -1, -1
   len_cols = len(data[0]) - 1
   if random_subset == True:
      cols = np.random.choice(len_cols, int(np.sqrt(len_cols)), replace=False)
   else:
      cols = range(len(data[0]) - 1)
   for col in cols:                             # Check every column but last
      for test_val in np.unique(data[:,col]):   # Check all values in that column
         Q = Question(col, test_val, feature_names)
         left, right = partition(data, Q)
         
         if left is None or right is None:
            continue
         elif min(len(left), len(right)) <=  min_samples_leaf:
            continue
         else:
            info = info_gain(left, right, gini(data))
            if info >= max_info:
               max_info = info         # If it maximizes information, replace it!
               best_col = col
               best_val = test_val
   return max_info, Question(best_col, best_val, feature_names)


# Problem 4
class Leaf:
   """Tree leaf node
   Attribute:
      prediction (dict): Dictionary of labels at the leaf"""
   def __init__(self,data):
      dict_list = []
      for label in np.unique(data[:,-1]):
         keep = data[data[:,-1] == label]
         dict_list.append((int(label), len(keep)))
      self.prediction = dict(dict_list)
leaf = Leaf(animals)

class Decision_Node:
   """Tree node with a question
   Attributes:
      question (Question): Question associated with node
      left (Decision_Node or Leaf): child branch
      right (Decision_Node or Leaf): child branch"""
   def __init__(self, question, right_branch, left_branch):
      self.question = question
      self.left = left_branch
      self.right = right_branch


## Code to draw a tree
def draw_node(graph, my_tree):
   """Helper function for drawTree"""
   node_id = uuid4().hex
   #If it's a leaf, draw an oval and label with the prediction
   if isinstance(my_tree, Leaf):
      graph.node(node_id, shape="oval", label="%s" % my_tree.prediction)
      return node_id
   else: #If it's not a leaf, make a question box
      graph.node(node_id, shape="box", label="%s" % my_tree.question)
      left_id = draw_node(graph, my_tree.left)
      graph.edge(node_id, left_id, label="T")
      right_id = draw_node(graph, my_tree.right)    
      graph.edge(node_id, right_id, label="F")
      return node_id

def draw_tree(my_tree):
   """Draws a tree"""
   #Remove the files if they already exist
   for file in ['Digraph.gv','Digraph.gv.pdf']:
      if os.path.exists(file):
         os.remove(file)
   graph = graphviz.Digraph(comment="Decision Tree")
   draw_node(graph, my_tree)
   graph.render(view=True) #This saves Digraph.gv and Digraph.gv.pdf

# Prolem 5
def build_tree(data, feature_names, min_samples_leaf=5, max_depth=4, current_depth=0, random_subset=False):
   """Build a classification tree using the classes Decision_Node and Leaf
   Parameters:
      data (ndarray)
      feature_names(list or array)
      min_samples_leaf (int): minimum allowed number of samples per leaf
      max_depth (int): maximum allowed depth
      current_depth (int): depth counter
      random_subset (bool): whether or not to train on a random subset of features
   Returns:
      Decision_Node (or Leaf)"""
   if current_depth < max_depth:
      info_gain, Q = find_best_split(data, feature_names, min_samples_leaf=min_samples_leaf, random_subset=random_subset)
      left, right = partition(data, Q)
      
      # Check if partition did nothing
      if left is None or right is None:
         return Leaf(data)

      # Check if the left or right is too small to split again
      if len(right) < min_samples_leaf:
         return Leaf(right) 
      if len(left) < min_samples_leaf:
         return Leaf(left)
      
      # Check if we gained any information
      if info_gain == 0:
         return Leaf(data)

      return Decision_Node(Q, build_tree(right, feature_names, current_depth=current_depth+1, 
                                         min_samples_leaf=min_samples_leaf, max_depth=max_depth),
                              build_tree(left, feature_names, current_depth=current_depth+1, 
                                         min_samples_leaf=min_samples_leaf, max_depth=max_depth))
      
   else:
      return Leaf(data)
   
   
# Problem 6
def predict_tree(sample, my_tree):
   """Predict the label for a sample given a pre-made decision tree
   Parameters:
      sample (ndarray): a single sample
      my_tree (Decision_Node or Leaf): a decision tree
   Returns:
      Label to be assigned to new sample"""  
   try:
      if my_tree.question.match(sample):
         return predict_tree(sample, my_tree.right)
      else:
         return predict_tree(sample, my_tree.left)
   except:
      pred_dict = my_tree.prediction
      total = sum(pred_dict.values())
      keys = [k for k,v in pred_dict.items()]
      probabilities = [v/total for k,v in pred_dict.items()]
      prediction = np.random.choice(keys, p=probabilities)
      return prediction

    
def analyze_tree(dataset,my_tree):
   """Test how accurately a tree classifies a dataset
   Parameters:
      dataset (ndarray): Labeled data with the labels in the last column
      tree (Decision_Node or Leaf): a decision tree
   Returns:
      (float): Proportion of dataset classified correctly"""
   correct = 0
   # Calculate probabilities based off dictionary for each row
   for row in dataset:
      prediction = predict_tree(row[:-1], my_tree)
      # Add up correct probabilities!
      if prediction == row[-1]:
         correct += 1
   return correct / len(dataset)





# Problem 7
def predict_forest(sample, forest):
   """Predict the label for a new sample, given a random forest
   Parameters:
      sample (ndarray): a single sample
      forest (list): a list of decision trees
   Returns:
      Label to be assigned to new sample"""
   predictions = []
   # Predict for each tree
   for tree in forest:
      prediction = predict_tree(sample, tree)
      predictions.append(prediction)
   # Find majority vote
   counts = np.bincount(predictions)
   return np.argmax(counts)



def analyze_forest(dataset,forest):
   """Test how accurately a forest classifies a dataset
   Parameters:
      dataset (ndarray): Labeled data with the labels in the last column
      forest (list): list of decision trees
   Returns:
      (float): Proportion of dataset classified correctly"""
   correct = 0
   total = 0
   # Calculate probabilities based off dictionary for each row
   for row in dataset:
      prediction = predict_forest(row[:-1], forest)
      # Add up correct probabilities!
      if prediction == row[-1]:
         correct += 1
   return correct / len(dataset)


# Problem 8
def prob8():
   """Use the file parkinsons.csv to analyze a 5 tree forest.
   
   Create a forest with 5 trees and train on 100 random samples from the dataset.
   Use 30 random samples to test using analyze_forest() and SkLearn's 
   RandomForestClassifier.
   
   Create a 5 tree forest using 80% of the dataset and analzye using 
   RandomForestClassifier.
   
   Return three tuples, one for each test.
   
   Each tuple should include the accuracy and time to run: (accuracy, running time) 
   """
   patients = np.loadtxt('parkinsons.csv', delimiter=',')
   features = np.loadtxt('parkinsons_features.csv', delimiter=',', dtype=str,
                           comments=None)[1:]

   np.random.shuffle(patients)
   train_set = patients[:100, 1:]
   test_set = patients[100:130, 1:]
   start = time.time()
   forest = []
   for j in range(5):      # Create forest
      test_tree = build_tree(train_set, features, min_samples_leaf=15, random_subset=True)
      forest.append(test_tree)
   acc = analyze_forest(test_set, forest)
   stop = time.time()
   elapsed_time_0 = stop-start

   # Sklearn with 100 samples
   start = time.time()
   forest1 = RandomForestClassifier(n_estimators=5, max_depth=4, min_samples_leaf=15)
   forest1.fit(train_set[:,:-1], train_set[:,-1])
   score1 = forest1.score(test_set[:,:-1], test_set[:,-1])
   stop = time.time()
   elapsed_time_1 = stop-start

   # Sklearn with all the samples
   train_80 = patients[:int(len(patients)*.8)]
   test_20 = patients[int(len(patients)*.8):]
   start = time.time()
   forest2 = RandomForestClassifier(n_estimators=5, max_depth=4, min_samples_leaf=15)
   forest2.fit(train_80[:,1:-1], train_80[:,-1])
   score2 = forest1.score(test_20[:,1:-1], test_20[:,-1])
   stop = time.time()
   elapsed_time_2 = stop-start

   return (acc, elapsed_time_0), (score1, elapsed_time_1), (score2, elapsed_time_2)