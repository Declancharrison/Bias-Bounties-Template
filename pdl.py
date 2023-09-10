#!/usr/bin/python3

#imports section
###
import numpy as np
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import accuracy_score as ACC
from sklearn.metrics import top_k_accuracy_score as TOP_K

import os
import dill as pkl
import shutil
import copy
import sys
import time
# sys.setrecursionlimit(10**9)
def load_model(folder):
    with open(f"{folder}/initial_model.pkl", "rb") as file:
        base_model = pkl.load(file)
    pdl_reg = PointerDecisionList(base_model, None, None, None, None, alpha = 1, min_group_size = 1, reload = 1)
    pdl_reg.reload_model(folder)
    return pdl_reg

class node:
    '''
    Substructure of the UDT class. Stores instructions for predicting on an instance within a level set at a given point in time for the UDT structure.

    Inputs:
        - self: self
        - right_child: next node in UDT sequence (node)
        - node_name: node name (str)
    '''
    def __init__(self, index, right_child=None, left_child = None, node_name='node'):
        # index
        self.index = index

        #node relation
        self.right_child = right_child
        self.left_child = left_child
        self.node_name = node_name

        # node data
        self.data = None

class addGroupNode:
    def __init__(self, index, right_child = None, left_child = None):
        #node location relation
        self.right_child = right_child
        self.left_child = left_child

        self.node_name = 'addGroupNode'

        #group function
        self.index = index

class updateNode:
    def __init__(self, index, right_child = None, left_child = None):
        #node location relation
        self.right_child = right_child
        self.left_child = left_child
        #update indicator
        self.node_name = 'updateNode'
        
        #group to update
        self.index = index

class repairNode:
    def __init__(self, index, pointer, right_child = None, left_child = None):
        #node location relation
        self.right_child = right_child
        self.left_child = left_child

        #update indicator
        self.node_name = 'repairNode'
        
        #group to update
        self.index = index

        # node to point to 
        self.pointer = pointer

        # node data
        self.data = None

class PointerDecisionList:
    '''
    The UDT structure is a list of nodes which each contain conditional statements for predicting on instances within a level set.

    Inputs:
        - self: self
        - T: 
    '''
    def __init__(self, initial_model, x_train, y_train, x_val, y_val, alpha, min_group_size, loss_fn_name = None, reload = -1):
        
        #set initialization
        self.head_node = node(-1, node_name = 'head node')
        self.tail_node = self.head_node
        self.current_node = self.head_node
        self.node_list = [self.head_node]
        self.min_group_size = min_group_size
        self.head_node_name = 'head node'
        self.initial_model = initial_model
        self.alpha = alpha
        self.updates = 0
        self.repairs = 0
        self.total_repairs = 0

        self.loss_fn_name = loss_fn_name

        if self.loss_fn_name == "MAE":
            self.loss_fn = MAE
        elif self.loss_fn_name == "MSE":
            self.loss_fn = MSE
        elif self.loss_fn_name == "ACC":
            self.loss_fn = ACC
        elif self.loss_fn_name == "TOP_K":
            self.loss_fn = TOP_K

        if reload == 1:
            return
        
        #initial predictions
        self.train_predictions = initial_model.predict(x_train)
        self.val_predictions = initial_model.predict(x_val)

        self.model_error_train = self.loss_fn(y_train, self.train_predictions)
        self.model_error_val = self.loss_fn(y_val, self.val_predictions)

        self.train_list = [self.model_error_train]
        self.val_list = [self.model_error_val]

        self.group_indices_train = {}
        self.group_indices_val = {}

        self.best_group_predictions_train = {}
        self.best_group_predictions_val = {}

        self.best_group_errors_val = np.array([])
        self.group_weights = np.array([])
        self.best_node_location = {}

        self.group_functions = []
        self.hypothesis_functions = []
    def append_node(self, new_node):
        self.tail_node.right_child = new_node
        new_node.left_child = self.tail_node
        self.tail_node = new_node
        self.node_list.append(new_node)

    def add_group(self, group, hypothesis, x_train, x_val):
        indices_train = group(x_train).astype('bool')
        indices_val = group(x_val).astype('bool')
        self.group_indices_train[self.updates] = np.array(indices_train)
        self.group_indices_val[self.updates] = np.array(indices_val)
        self.group_functions.append(group)
        self.hypothesis_functions.append(hypothesis)
        self.group_weights = np.append(self.group_weights, indices_val.sum()/len(indices_val))
        self.append_node(addGroupNode(self.updates))

    def compute_group_errors(self, y_train, y_val):
        #comput group MSE
        self.group_errors_train = []
        self.group_errors_val = []

        for index in range(len(self.group_indices_train)):
            group_indices = self.group_indices_train[index]
            group_mse_train = self.loss_fn(y_train[group_indices],self.train_predictions[group_indices])
            self.group_errors_train.append(group_mse_train)

        for index in range(len(self.group_indices_val)):
            group_indices = self.group_indices_val[index]
            group_mse_val = self.loss_fn(y_val[group_indices],self.val_predictions[group_indices])
            self.group_errors_val.append(group_mse_val)
        
        self.group_errors_train = np.array(self.group_errors_train)
        self.group_errors_val = np.array(self.group_errors_val)

    def update_group_predictions(self):

        #add group best predictions node
        update_array = (self.group_errors_val < self.best_group_errors_val)
        for index in range(len(update_array)):
            if update_array[index] == True:
                #not actually best train preds but for naming conventions sake
                self.best_group_predictions_train[index] = self.train_predictions[self.group_indices_train[index]]
                self.best_group_predictions_val[index] = self.val_predictions[self.group_indices_val[index]]
                self.best_node_location[index] = len(self.node_list) - 1

        if update_array.sum() != 0:
            self.append_node(updateNode(np.where(update_array == True)[0]))
        
        #update best group errors
        np.putmask(self.best_group_errors_val, self.best_group_errors_val > self.group_errors_val, self.group_errors_val)
   
    def check_group_violations(self):
        #check for group error violations
        violations_array = (self.group_weights * (self.group_errors_val - self.best_group_errors_val) > self.alpha)
        for index in range(len(violations_array)):
            if violations_array[index] == True:
                self.repairs += 1
                self.total_repairs += 1
                self.append_node(repairNode(index, self.best_node_location[index]))
                replace_indices = self.group_indices_train[index]
                self.train_predictions[replace_indices] = self.best_group_predictions_train[index]
                replace_indices = self.group_indices_val[index]
                self.val_predictions[replace_indices] = self.best_group_predictions_val[index]
                return True
        return False

    def repair_groups(self, y_train, y_val):
        #check if there are groups to protect
        if len(self.group_indices_val) != 0:
            violated_group = True
            while violated_group == True:
                self.compute_group_errors(y_train, y_val)
                
                self.update_group_predictions()

                violated_group = self.check_group_violations()
        return

    def update(self, group, hypothesis, x_train, y_train, x_val, y_val):
        
        indices_val = group(x_val).astype('bool')

        if indices_val.sum() <= self.min_group_size:
            return False

        model_error_val = self.loss_fn(y_val[indices_val], self.val_predictions[indices_val])
        
        hypothesis_preds_val = hypothesis(x_val[indices_val])

        hypothesis_error_val = self.loss_fn(y_val[indices_val], hypothesis_preds_val)

        self.repairs = 0

        if ( ( (indices_val.sum()/len(indices_val)) *(model_error_val - hypothesis_error_val) ) > self.alpha):
            #metrics for paper, lightweight version would not require the following
            indices_train = group(x_train).astype('bool')
            hypothesis_preds_train = hypothesis(x_train[indices_train])
            self.add_group(group, hypothesis, x_train, x_val)
            self.append_node(node(self.updates, right_child=None, node_name='node'))
            self.best_node_location[self.updates] = len(self.node_list) - 1

            self.train_predictions[indices_train] = hypothesis_preds_train
            self.val_predictions[indices_val] = hypothesis_preds_val
            self.best_group_errors_val = np.append(self.best_group_errors_val, self.loss_fn(self.y_val[indices_val], self.val_predictions[indices_val]))
            self.best_group_predictions_train[self.updates] = self.train_predictions[indices_train]
            self.best_group_predictions_val[self.updates] = self.val_predictions[indices_val]
            
            self.repair_groups(y_train, y_val)

            self.train_list.append(self.loss_fn(y_train, self.train_predictions))
            self.val_list.append(self.loss_fn(y_val, self.val_predictions))

            self.updates += 1
            # print("Update Accepted!")
            return True
        
        # print("Update Rejected!")
        return False


    def predict(self, X):

        predictions = np.array([None]*len(X), dtype = float)

        current_indices_allowable = np.ones(len(X)).astype('bool')

        group_indices_X = {}

        self.current_node = self.tail_node
        
        data_empty = np.zeros(len(X)).astype('bool')

        total_number_predictions = len(X)

        predictions_count = 0
        # get to the first non group/update node. 

        data_empty = np.zeros(len(X)).astype('bool')
        while self.current_node.node_name in ['addGroupNode', 'updateNode']:
            self.current_node = self.current_node.left_child
            continue

        # while self.current_node != self.head_node and (has_prediction == 0).any():
        while (self.current_node != self.head_node):

            # check if we can leave early
            if (predictions_count == total_number_predictions):
                break

            # check if it is a node we don't care about
            if self.current_node.node_name in ['addGroupNode', 'updateNode']:
                self.current_node = self.current_node.left_child
                continue
            
            # get node group indices
            try: 
                group_indices = group_indices_X[self.current_node.index]
            except:
                group = self.group_functions[self.current_node.index]
                group_indices = group(X)
                group_indices_X[self.current_node.index] = group_indices

            if type(self.current_node.data) == type(None):
                data_bool = data_empty
            else:
                data_bool = self.current_node.data

            current_indices_allowable = current_indices_allowable | data_bool

            if self.current_node.node_name == 'repairNode':
                forward_indices = current_indices_allowable & group_indices 
                if type(self.node_list[self.current_node.pointer].data) == type(None):
                    self.node_list[self.current_node.pointer].data = forward_indices
                else:
                    self.node_list[self.current_node.pointer].data = (self.node_list[self.current_node.pointer].data | forward_indices)
                current_indices_allowable[forward_indices] = False
            else:
                update_indices = group_indices & current_indices_allowable
                try:
                    predictions[update_indices] = self.hypothesis_functions[self.current_node.index](X[update_indices])
                    current_indices_allowable[update_indices] = False
                    predictions_count += update_indices.sum()
                except:
                    pass
                

            self.current_node.data = None

            self.current_node = self.current_node.left_child

        if (current_indices_allowable).any():
            predictions[current_indices_allowable] = self.initial_model.predict(X[current_indices_allowable])
        
        return np.array(predictions, dtype = float)

    def track(self, X, y, return_predictions = False):
        
        error_list = []
        historical_predictions = []
        self.current_node = self.head_node

        predictions = self.initial_model.predict(X)    

        historical_predictions.append(copy.deepcopy(predictions))
        error_list.append(self.loss_fn(y, predictions))

        best_group_predictions = np.empty((0,len(predictions)), float)

        group_list = np.empty((0,len(predictions)), bool)

        if self.current_node.right_child != None:
            self.current_node = self.current_node.right_child
        else:
            return error_list

        #traverse down UDT
        while self.current_node != None:
            #group prediction updates
            if 'updateNode' == self.current_node.node_name:
                for index in self.current_node.index:
                    best_group_predictions[index] = copy.deepcopy(predictions)
                self.current_node = self.current_node.right_child
                continue

            if 'addGroupNode' == self.current_node.node_name:
                group_list = np.vstack([group_list, np.array(self.group_functions[self.current_node.index](X))])
                self.current_node = self.current_node.right_child
                continue

            #group prediction repairs
            if 'repairNode' == self.current_node.node_name:
                group_indices = group_list[self.current_node.index]
                predictions[group_indices] = best_group_predictions[self.current_node.index][group_indices]
                self.current_node = self.current_node.right_child
                error_list[-1] = (self.loss_fn(y, predictions))
                historical_predictions[-1] = copy.deepcopy(predictions)
                continue
        
            index_updates = self.group_functions[self.current_node.index](X)
            
            if index_updates.sum() == 0:
                self.current_node = self.current_node.right_child
                best_group_predictions = np.vstack([best_group_predictions, copy.deepcopy(predictions)])
                error_list.append(self.loss_fn(y, predictions))
                historical_predictions.append(copy.deepcopy(predictions))
                continue

            node_predictions = self.hypothesis_functions[self.current_node.index](X[index_updates])

            np.put(predictions, np.where(index_updates == 1), node_predictions)

            best_group_predictions = np.vstack([best_group_predictions, copy.deepcopy(predictions)])

            self.current_node = self.current_node.right_child

            error_list.append(self.loss_fn(y, predictions))
            
            historical_predictions.append(copy.deepcopy(predictions))

        if return_predictions == False:
            return error_list
        else:
            return error_list, historical_predictions

    def save_model(self, directory_name = 'PDL'):
        # sys.setrecursionlimit(10**9)

        state_dict = {"alpha": self.alpha, 
                      "updates": self.updates,
                      "train_predictions" : self.train_predictions,
                      "val_predictions" : self.val_predictions,
                      "model_error_train" : self.model_error_train,
                      "model_error_val": self.model_error_val,
                      "train_list": self.train_list,
                      "val_list": self.val_list,
                      "group_weights": self.group_weights,
                      "group_indices_train": self.group_indices_train,
                      "group_indices_val": self.group_indices_val,
                      "best_group_predictions_train": self.best_group_predictions_train,
                      "best_group_predictions_val": self.best_group_predictions_val,
                      "best_group_errors_val": self.best_group_errors_val,
                      "best_node_location": self.best_node_location,
                      "total_repairs": self.total_repairs,
                      "repairs": self.repairs,
                      "loss_fn_name": self.loss_fn_name
                    }

        if os.path.exists(directory_name):
            shutil.rmtree(directory_name)
        os.mkdir(directory_name)
        os.mkdir(f'{directory_name}/groups')
        os.mkdir(f'{directory_name}/hypotheses')

        for index, group in enumerate(self.group_functions): 
            pkl.dump(group, open(f'{directory_name}/groups/g{index}.pkl', 'wb'))

        for index, hypothesis in enumerate(self.hypothesis_functions):
            pkl.dump(hypothesis, open(f'{directory_name}/hypotheses/h{index}.pkl', 'wb'))

        pkl.dump(self.initial_model, open(f'{directory_name}/initial_model.pkl', 'wb'))

        node = self.head_node.right_child

        node_dict = {}
        counter = 0

        while node != None:

            if node.node_name == "repairNode":
                node_dict[counter] = [node.node_name, node.index, node.pointer]
            else:
                node_dict[counter] = [node.node_name, node.index]
            node = node.right_child
            counter += 1
        state_dict["node_dict"] = node_dict
        pkl.dump(state_dict, open(f'{directory_name}/state_dict.pkl', 'wb'))

        return
    
    def reload_model(self, directory_name = 'PDL'):

        self.group_functions = []
        self.hypothesis_functions = []

        state_dict = pkl.load(open(f'{directory_name}/state_dict.pkl', 'rb'))

        if ((len(self.node_list) != 1) and (len(state_dict["train_predictions"]) != len(self.train_predictions)) and (len(state_dict["val_predictions"]) != len(self.val_predictions))):
            print("ERROR")
            return
        
        self.alpha = state_dict["alpha"]
        self.updates = state_dict["updates"] 
        self.train_predictions = state_dict["train_predictions"]  
        self.val_predictions = state_dict["val_predictions"]  
        self.model_error_train = state_dict["model_error_train"] 
        self.model_error_val = state_dict["model_error_val"] 
        self.train_list = state_dict["train_list"] 
        self.val_list = state_dict["val_list"] 
        self.group_weights = state_dict["group_weights"]
        self.group_indices_train = state_dict["group_indices_train"] 
        self.group_indices_val = state_dict["group_indices_val"] 
        self.best_group_predictions_train = state_dict["best_group_predictions_train"] 
        self.best_group_predictions_val = state_dict["best_group_predictions_val"] 
        self.best_group_errors_val = state_dict["best_group_errors_val"] 
        self.best_node_location = state_dict["best_node_location"] 
        self.total_repairs = state_dict["total_repairs"]
        self.repairs = state_dict["repairs"]
        self.loss_fn_name = state_dict["loss_fn_name"]

        if self.loss_fn_name == "MAE":
            self.loss_fn = MAE
        elif self.loss_fn_name == "MSE":
            self.loss_fn = MSE
        elif self.loss_fn_name == "ACC":
            self.loss_fn = ACC
        elif self.loss_fn_name == "TOP_K":
            self.loss_fn = TOP_K

        for counter in range(len(state_dict["node_dict"])):
            name = state_dict["node_dict"][counter][0]
            index = state_dict["node_dict"][counter][1]
            if name == "repairNode":
                pointer = state_dict["node_dict"][counter][2]
                self.append_node(repairNode(index, pointer))
            
            if name == "addGroupNode":
                self.append_node(addGroupNode(index))

            if name == "node":
                self.append_node(node(index))

            if name == "updateNode":
                self.append_node(updateNode(index))

        self.initial_model = pkl.load(open(f'{directory_name}/initial_model.pkl', 'rb'))

        for index in range(self.updates):
            group = pkl.load(open(f'{directory_name}/groups/g{index}.pkl', 'rb'))
            try:
                group.__self__.n_jobs = -1
            except:
                pass
            self.group_functions.append(group)

            hypothesis = pkl.load(open(f'{directory_name}/hypotheses/h{index}.pkl', 'rb'))
            try:
                hypothesis.__self__.n_jobs = -1
            except:
                pass
            self.hypothesis_functions.append(hypothesis)
        return 
