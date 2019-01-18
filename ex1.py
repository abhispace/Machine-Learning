import numpy as np
#import scipy as sp
import matplotlib.pyplot as plt
#import csv

#Feature Normalisation:
def norm(x):
    stdv_x=np.std(x)
    x=(x-np.mean(x))/stdv_x
    return x
    
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

#Sample data from Ex1 course
input_file_loc = "/abhinav/Documents/machine_learning/data/ml_an/"
file2 = open(input_file_loc+"ex1data1.txt","r")
dataex1 = np.genfromtxt(file2,delimiter=',')
file2.close()

#Theta is the parameter to fit
theta = np.squeeze(np.zeros((2, 1)))                                 # initialize fitting parameters
#Gradient descent setttings
iterations = 1500
alpha = 0.01

xdata=np.column_stack((np.ones(len(dataex1[:,1])),dataex1[:,0]))
ydata=dataex1[:,1]

# Expected cost value (approx) 32.07
cost=computecost(xdata,ydata,theta)
print "cost = ", cost, "with theta = ", theta

#More testing:: Expected cost value (approx) 54.24
newtheta=np.array([-1,2])
costnew=computecost(xdata,ydata,newtheta)
print "cost = ", costnew, "with theta = ", newtheta

# Run gradient descent
theta,cost_hist = gradientdescent(xdata, ydata, theta, alpha, iterations)
print "Theta after gradient descent= ", theta

# Plot the linear fit on top of data
plt.scatter(xdata[:,1],ydata)
plt.plot(xdata[:,1],np.matmul(xdata,theta))
plt.show()