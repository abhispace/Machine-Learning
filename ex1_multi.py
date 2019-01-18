import numpy as np
#import scipy as sp
import matplotlib.pyplot as plt
#import csv

#Feature Normalisation:
def norm(x):
    mean=np.array([np.mean(x[:,0]),np.mean(x[:,1])])
    stdv=np.array([np.std(x[:,0]),np.std(x[:,1])])
    x=(x-mean)/stdv
#    print mean,stdv
    return x,mean,stdv
    
#COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
#   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
#   parameter for linear regression to fit the data points in X and y

def computecostmulti(x, y, theta):
    m = len(y)                                 # number of training examples
    cost = 0
    cost = (1/(2*m))*sum(np.square(np.matmul(x,theta)-y))
    return cost

def computecost(x, y, theta):
    m = len(y)                                     # number of training examples
    costis = 0
    costis = (1.0/(2.0*m))*sum(np.square(np.matmul(x,theta)-y))
    return costis


#GRADIENTDESCENT Performs gradient descent to learn theta
#   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
#   taking num_iters gradient steps with learning rate alpha

def gradientdescent(x, y, theta, alpha, num_iters):
    m = len(y)                                 # number of training examples
    cost_hist = np.zeros((num_iters, 1))
    for i in range(1,num_iters):
        theta=theta-(1.0/m)*alpha*np.matmul((np.transpose(x)),(np.matmul(x,theta)-y))
        cost_hist[i] = computecost(x, y, theta)
    return theta, cost_hist

#NORMALEQN Computes the closed-form solution to linear regression 
#   NORMALEQN(X,y) computes the closed-form solution to linear 
#   regression using the normal equations.
def normaleqn(x, y):
    theta = np.zeros(np.size(x,1))
    theta = np.matmul(np.linalg.pinv(np.matmul(np.transpose(x), x)), np.matmul(np.transpose(x), y))
    return theta


#Sample data from Ex1 course
input_file_loc = "/abhinav/Documents/machine_learning/data/ml_an/"
file2 = open(input_file_loc+"ex1data2.txt","r")
dataex1 = np.genfromtxt(file2,delimiter=',')
file2.close()

xdata=np.column_stack((dataex1[:,0],dataex1[:,1]))
ydata=dataex1[:,2]

#Normalise features
xdata, mu, sigma = norm(xdata)
#Add a column of ones to x data
xdata=np.column_stack((np.ones(len(xdata[:,0])),xdata))

#----------------------------GRADIENT DESCENT---------------------------------------------
#Theta is the parameter to fit
theta = np.squeeze(np.zeros((3, 1)))                                 # initialize fitting parameters
#Gradient descent setttings
iterations = 50
alpha = 0.1

theta, cost_hist = gradientdescent(xdata,ydata,theta,alpha,iterations)

plt.plot(cost_hist[1:])

# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
# Recall that the first column of X is all-ones. Thus, it does
# not need to be normalized.
test = np.array([1650,3])
test = (test-mu)/sigma
test2 = np.concatenate(([1], test), axis=None)
price = np.matmul(test2,theta); # You should change this


# ============================================================

print "Predicted price of a 1650 sq-ft, 3 br house using gradient descent=", price

## ================ Part 3: Normal Equations ================

print "Solving with normal equations..."

# ====================== YOUR CODE HERE ======================
# Instructions: The following code computes the closed form 
#               solution for linear regression using the normal
#               equations. You should complete the code in 
#               normalEqn.m
#
#               After doing so, you should complete this code 
#               to predict the price of a 1650 sq-ft, 3 br house.
#

## Load Data
xdata=np.column_stack((dataex1[:,0],dataex1[:,1]))
ydata=dataex1[:,2]
m = len(ydata);

# Add intercept term to X
xdata=np.column_stack((np.ones(len(xdata[:,0])),xdata))

# Calculate the parameters from the normal equation
thetanew = normaleqn(xdata, ydata)

# Display normal equation's result
print "Theta computed from the normal equations:", theta


# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
test = np.array([1650,3])
test = (test-mu)/sigma
test2 = np.concatenate(([1], test), axis=None)
price = np.matmul(test2,thetanew); # You should change this


# ============================================================

print "Predicted price of a 1650 sq-ft, 3 br house using normal equations: ", price

