import numpy as np
#import scipy as sp
import matplotlib.pyplot as plt
#import csv
import scipy.optimize as opt


#Feature Normalisation:
def norm(x):
    stdv_x=np.std(x)
    x=(x-np.mean(x))/stdv_x
    return x
    
def sigmoid(x):
    return 1./(1.0+np.exp(-1.0*x))
    
def computecost(theta,x, y):
    m = len(y)                                     # number of training examples
    costis = 0
    g=np.zeros(np.size(theta))
    trans_theta=np.transpose(theta)
    htheta=sigmoid(np.matmul(x,trans_theta))
    log_htheta=np.log(htheta)
    costis = (-1.0/m)*sum(np.multiply(y,log_htheta)+np.multiply((1.0-y),np.log(1.0-htheta)))
    return costis

def gradient (theta,x,y):
    m = len(y)                                     # number of training examples
    g=np.zeros(np.size(theta))
    trans_theta=np.transpose(theta)
    trans_x=np.transpose(x)
    htheta=sigmoid(np.matmul(x,trans_theta))
    g = (1./m)*np.matmul(trans_x,(htheta-y))
    return g

def predict(theta,x):
    p = np.round(sigmoid(np.matmul(x,theta)))
    return p

#Sample data from Ex1 course
input_file_loc = "/abhinav/Documents/machine_learning/data/ml_an/"
file2 = open(input_file_loc+"ex2data1.txt","r")
dataex1 = np.genfromtxt(file2,delimiter=',')
file2.close()

xdata=np.column_stack((dataex1[:,0],dataex1[:,1]))
ydata=dataex1[:,2]

x_positive=np.where(ydata==1)
x_negative=np.where(ydata==0)


#Plotting the data to visualise the problem
'''
plt.scatter(xdata[x_positive,0],xdata[x_positive,1],marker="o",label="admitted")
plt.scatter(xdata[x_negative,0],xdata[x_negative,1],marker="s",label="rejected")
plt.xlabel("exam 1 score")
plt.ylabel("exam 2 score")
plt.legend(loc='upper right')
plt.show()
'''

xdata=np.column_stack((np.ones((np.size(xdata,0))),dataex1[:,0],dataex1[:,1]))
initial_theta=np.zeros((np.size(xdata,1)))

# Expected cost value (approx) 32.07
cost=computecost(initial_theta,xdata,ydata,)
grad=gradient(initial_theta,xdata,ydata)
print "cost = ", cost, "with theta = ", initial_theta
print "gradient = ", grad

test_theta=np.array([-24.0, 0.2, 0.2])
cost= computecost(test_theta,xdata,ydata)
grad=gradient(test_theta,xdata,ydata)
print "cost = ", cost, "with theta = ", test_theta
print "gradient = ", grad

result = opt.fmin_tnc(func=computecost, x0=initial_theta, fprime=gradient, args=(xdata, ydata))
cost=computecost(result[0], xdata, ydata)
theta=result[0]
print "optimum theta = ",result[0], "cost = ", cost

# Plot the linear fit on top of data
x_loc=np.array([np.min(xdata[:,1])-2., np.max(xdata[:,1])+2.])
y_loc=(-1./theta[2])*(theta[1]*x_loc+theta[0])
plt.scatter(xdata[x_positive,1],xdata[x_positive,2],marker="o",label="admitted")
plt.scatter(xdata[x_negative,1],xdata[x_negative,2],marker="s",label="rejected")
plt.plot(x_loc,y_loc,c="g")
plt.xlabel("exam 1 score")
plt.ylabel("exam 2 score")
plt.legend(loc='upper right')
plt.show()


#Predict and Accuracies
# We use the logarithmic regression to predict the probability
# that a student will be admitted

ex1_score=45.
ex2_score=85.
prob=sigmoid(np.matmul(np.array([1., ex1_score, ex2_score]),theta))
print "Scores: ",ex1_score," and ",ex2_score
print "Probability of admission: ", prob

# Compute accuracy on training set
p = predict(theta,xdata)
accuracy=np.mean(p==ydata)*100.
print "Train accuracy: ", accuracy