import numpy as np
from random import random
from ard import arange_data

import pandas as pd

df = pd.read_csv('smoke_detection_iot.csv')

""" This code implements NN algorithm for smoke detection"""

#################
#################
#################-

# IGNORE BELOW PART #

# def NeuralNet(input_layer,hidden_layers,output_layer):
    
#     # We have to express the neural network in terms of the parameters of the function.
    
#     NN=[] # This list will be something like NN=[3,4,5,2], for example, for input_layer=3,
#           # hidden_layers=[4,5], and output_layer=2.
    
#     NN.append(input_layer)
#     NN.extend(hidden_layers) # The elements of "hidden_layers" will be stored in "NN" list.
#     NN.append(output_layer)
    
#     return NN

# # Note that above function constructs the Neural Network.

# # Start the neural network.

# NN=NeuralNet(14,[7,7,7,7],1)

# Define the initial values of the weights between the layers, derivatives of the loss function w.r.t
# weights, and the activation in the input layer, i.e. the feature vectors in the dataset.

###############
###############
###############

# Define the forward propogation.


def forward_propogation(x_i,weight_matrix,NN): # x_i's are the activations.
    global v_list # To monitor the v_i's.      # in the input layer.
    activations=[x_i] # Note that the activations in the input layers are just the feature vectors.    
    v_list=[]                         
                     
    for i in range(len(NN)-1): # Forward propogation is done in matrix form via NN equations.
       
        v_i= np.dot(x_i,weight_matrix[i])
        
        v_list.append(v_i)
    
        
        
        x_i=sigmoid(v_i) # These are the activations.
        
        
        activations.append(x_i) # Save the activation values at each layer.
    
    return activations

# Initialize the function above to get the activation values up to the output layer.

# Define the back propogation.

# Note that "loss" input is the error (y-y_hat) in the output layer.

def back_propogation(loss,activations,weight_matrix,NN):
    global loss_list # For monitoring.
    global derivative_list # For monitoring the gradients, which is crucial.
    derivative_list=[]
    loss_list=[]
    for i in reversed(range(len(NN)-1)): # List is reversed since we start from the last layer.
        
        # "Delta"s are the deltas in the NN equations.
        
        delta=loss*first_derivative_of_sigmoid(v_list[i])
        
        delta_reshaped = delta.reshape(delta.shape[0], -1).T # We reshape it to have a column vector structure.
        
        activation=activations[i] # "activation" is the activation value in the i+2'th layer.
         
        activations_reshaped=activation.reshape(activation.shape[0], -1) # We reshape it to have a column vector structure.
        
        derivative_list.append(np.dot(activations_reshaped,delta_reshaped))
        
        loss=np.dot(delta,weight_matrix[i].T)
        
        loss_list.append(loss)
        

    return derivative_list[::-1] # We return the reversed derivative_list to match the shape with weight_matrix in the Gradient Descent method.
    
# Define the Gradient Descent method.

def GD(n,weight_matrix,derivative_list): # n is the learning rate.
    global a,b # For monitoring weight matrix and derivative_list inside the Gradient Descent.
    for i in range(len(weight_matrix)):
        a=weight_matrix
        
        b=derivative_list

        weight_matrix[i]+=derivative_list[i]*n # Gradient Descent is done here via the gradients found by backpropogation.
        
    return weight_matrix # Return the updated weights.
        
        
    
        
        
# Use Sigmoid function both in the hidden and output layers.


def sigmoid(x):

    return 1/(1+np.exp(-x))

def first_derivative_of_sigmoid(x):
    
    return (1-sigmoid(x))*sigmoid(x)


# Define the mean square error (for n=1 obviously).
def mse(y,y_hat):
    return (y-y_hat)**2

# Define the training function.      

def train(x_train,y_train,epoch,weight_matrix,n,NN): # Epoch is the number of steps and n is the learning rate.
    

    for i in range(epoch):
     
        sum_errors = 0 # For finding the total training error.
        
        for j, inputs in enumerate(x_train): # Iterate over training data.
      
            
            y_j=y_train[j] # Get the fire alarm values of the training data.
            
            activations=forward_propogation(inputs,weight_matrix,NN) # Forward propogate the training data.
            
            loss=y_j-activations[-1] # Calculate the error yielded in the output layer for initial weights.
            
            grad=back_propogation(loss,activations,weight_matrix,NN) # Back propogate the training data.
            
            weight_matrix=GD(n,weight_matrix,grad) # Update the weights.
            
            sum_errors +=mse(y_j, activations[-1]) # Find the mean square training error.
        
        # Inform us with at which epoch value we are.
        if i%50==0:  
            print("Mean Square Error: {} at epoch {}".format(sum_errors / len(x_train), i+1))

    print("Training is complete!!!!")
    print("-----------------")
    return weight_matrix # Return the updated weights for the trained model.
    


def predict(x_test,weight_matrix,NN): 
    predicted_labels = []
    for i, inputs in enumerate(x_test):
        act = forward_propogation(inputs, weight_matrix,NN)
        prediction = act[-1] #Final Prediction is the activation in the output layer.
        
        # Set a threshold in the output_layer above which you label the predicted label as 1 and below which you label 0.
        if prediction > 0.5:
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)
    return predicted_labels # Return the predicted labels.

# Below function computes the test error.
        
def calculate_error_test(y, y_pred):
    
    error = 0
    for i in range (len(y)):
        
        if y[i] != y_pred[i]: # if the predicted label does not match the true label, add error a value of 1.
            error = error + 1
    total_error = error/len(y)
    return total_error

# Below function is for Cross Validation to estimate the optimal NN structure and learning rate.

def CrossValidation(lr_list, structure_list,data,epoch,k_fold): # K-fold CV.

    error_global=[] # For storing the errors of each (lr_list,structure_list) pair.
    i_j_list=[]
    
    for i in lr_list:
        error_list=[]
     
        for j in structure_list:
            for k in range(1,k_fold):
                
                
                weight_matrix=[]
                for _ in range(len(j)-1):
                    weights=np.random.normal(0,1,size=(j[_],j[_+1]))
                    weight_matrix.append(weights)
                    
          
                x=data[0]
                y=data[1]
                
                x_test=x[(k-1)*int(len(x)/k_fold):k*int(len(x)/k_fold)] # Assume 500 test data.
                x_train=np.delete(x[0:int(len(x))],range((k-1)*int(len(x)/k_fold),k*int(len(x)/k_fold)),axis=0) # Assume 2000 train data.
                
                y_test=y[(k-1)*int(len(y)/k_fold):k*int(len(y)/k_fold)] # Assume 500 test data.
                y_train=np.delete(y[0:int(len(y))],range((k-1)*int(len(y)/k_fold),k*int(len(y)/k_fold)),axis=0)
                
                sol= train(x_train,y_train,epoch,weight_matrix,i,j)
                
                pred = predict(x_test, sol,j)

                error = calculate_error_test(y_test, pred)
            
                error_list.append(error) #
                
            i_j_list.append([i,j])
            error_global.append(sum(error_list)/k_fold) # Find the total error via sum function and divide it by the # of folds (i.e. k_fold) to find average error, then append it into the list.
        
    sorted_error=np.argsort(error_global) # Sort the average CV errors for each (lr_list,structure_list) pair by their indices so that we can easily access the index of the optimum k value.
    
    min_index=sorted_error[0]
    
    min_i_j=i_j_list[min_index]
    
    return min_i_j, error_global[min_index],error_global,i_j_list 

# Define the learning rates that we will iterate over in cross validation.

lr_list=[i*0.2 for i in range(1,6)]    

# Define the NN structures (for instance structure_list=[14,5,5,1] means there are 2 hidden layers with 5 neurons
# and 14 neurons in the input layer, 1 neuron in the output layer).
structure_list=[]
for i in range(1,6):
    liste = [13, i, 1]
    liste_1 = [13, i, i, 1]
    structure_list.append(liste)
    structure_list.append(liste_1)

# Take a slice of the data.

data=arange_data(df,20000)      

# We apply 5-fold CV with 100 epochs to reduce the computational cost.
epoch=200
k_fold=5

# Get the results of the NN algorithm.

result=CrossValidation(lr_list, structure_list,data,epoch,k_fold)
    
    
    
