
import pandas as pd
import numpy as np
from statistics import mode
import matplotlib.pyplot as plt
from scipy import interpolate


df = pd.read_csv('smoke_detection_iot.csv')
df.drop(['Unnamed: 0'], inplace = True, axis = 1)
df.head(n = 20)


# First lets convert data frame to design matrix.
design = df.to_numpy()
# design[:,2]

np.random.shuffle(design) # Shuffle the data once so that it is not "biased". What we mean by biased here is that
                          # original dataset contains full zeros (no smoke) for almost first 3000 data points.
                          # We do not want that since in this case KNN gives no error since a test data point is surrounded by all zeros.


# Finding maximum and minimum values of the feature vector elements.
maximum_values = []
minimum_values=[]
for i in range (len(df.columns.values.tolist())):
    minimum=np.min(design[:,i])
    maximum = np.max(design[:,i])
    maximum_values.append(maximum)
    minimum_values.append(minimum)

# Apply max-min scaling so that Euclidean distance is not dominated by a large term.
for i in range(len(df.columns.values.tolist())):
    design[:,i] = (design[:,i]-minimum_values[i])/(maximum_values[i]-minimum_values[i])

# Define a function that finds the Euclidean distance between test and training data points.
def ed(row0, row1):
    row0 = np.array(row0)
    row1 = np.array(row1)

    dist = np.sqrt(np.sum((row0-row1)**2))

    return dist # Returns the distance.

def KNN(x_train,x_test,y_train,y_test,k):
    error=0
    distance_list_all=[] # This list is used for storing distance lists for all test data.
    for test_row in x_test: # Iteration over test data.
        distance_list=[] # This list is used for storing distance list for each test data.
        for train_row in x_train: # Iteration over train data.
            dist=ed(train_row,test_row) # Euclidean distance between test and training data points.
            distance_list.append(dist)
        
        distance_list_all.append(distance_list)
    
    predicted_values_global=[] # This list is used for storing the predicted values for each test data.
    for i in range(len(distance_list_all)): # Note that len(distance_list_all)=len(x_test)
        
       
        indices=np.argsort(distance_list_all[i])
        indices=indices[0:k]# Sort by indices and choose the first k of them.
        k_nearest_values=[] # This list is used for storing the classes (0 or 1) of k-nearest neighboors of each test data.
        for j in indices: # Iterate over indices so that you can find which predicted value (0 or 1) that index corresponds to.
            k_nearest_values.append(y_train[j]) # Append the predicted value.
            
        most_common=mode(k_nearest_values) # Choose the most common class (0 or 1) among the k-nearest neighboors.
        predicted_values_global.append(most_common) 
        error+=abs(y_test[i]-most_common) # Error is found via 0-1 loss function.
    
    return 100*error/len(y_test) # Return the error for k-nearest neighboors.
        
   
def CrossValidation(k_fold): # K-fold CV.
    error_global=[] # This list is used for storing the mean k-fold error (total error/value of k_fold) for each K value of the k-nearest neighboor algorithm.
    for i in range(1,20,2): # Search the optimum k value (take it as an odd number).
        
        error=[] # This list is for each k value, which means it becomes empty for each k value.
        for j in range(1,k_fold+1): # We do the K-fold CV here. We go Test, Train, Train, Train, Train ; Train, Test, Train, Train, Train, ...; Train, Train, Train, Train, Test.
            print(j)
            test_data=design[(j-1)*1000:j*1000] # Assume 500 test data.
            train_data=np.delete(design[0:5000],range((j-1)*1000,j*1000),axis=0) # Assume 2000 train data.
            y_train=train_data[:, -1] # Split the predicted values (note they are in the last column).
            x_train=train_data[:, :-2] # Split the training data from the predicted values. Note that -2 is there because of the fact that column no. 14 had nothing to do with the data, thus we have erased it.
            y_test=test_data[:, -1] # Split the true values (they are in the last column).
            x_test=test_data[:, :-2] # Split the test data from the true values. -2 is there because of the same argument.
            error.append(KNN(x_train,x_test,y_train,y_test,i)) # Run KNN and find the error.
        error_global.append(sum(error)/k_fold) # Find the total error via sum function and divide it by the # of folds (i.e. k_fold) to find average error, then append it into the list.
        
    sorted_error=np.argsort(error_global) # Sort the average CV errors for each k value by their indices so that we can easily access the index of the optimum k value.
    
    return range(1,20,2)[sorted_error[0]],error_global# Return the optimum k value.

print(CrossValidation(5)) # Run CV.

# Below can be uncommented to produce the figure in the interim report.

# data=[0.05, 0.2, 0.2, 0.22000000000000003, 0.26, 0.3, 0.31999999999999995, 0.36, 0.33999999999999997, 0.3]
# x=range(1,20,2)
# xnew=range(1,20,2)
# f=interpolate.interp1d(x,data,kind="cubic")
# plt.plot(x,data,'o',xnew,f(xnew),'-')    
# plt.legend(['Data', 'Cubic Fit'], loc = 'best')
# plt.xlabel('K-Value')
# plt.ylabel('Error (Percentage)')
# plt.title('Error Percentage vs K-Value')

            
            
        
        
    
            
     
        
    
    
        
    
            
        

    
 
