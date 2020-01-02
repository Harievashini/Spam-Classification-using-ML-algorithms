from sklearn import metrics
from sklearn.metrics import *
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
import matplotlib.pyplot as plot
import numpy as np 
import math

#Part 1- Classification with Naive bayes
print "Gaussian Naive bayes"
#loading the dataset 
data = np.loadtxt('spambase.data.csv', delimiter=',')
#splitting into attributes and labels 
x = data[:,:57] #Attribute set
y=data[:,57] #Label set
#splitting into train and test data set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.50) 
# count of spam and non spam data in training set
nonspam_train = np.nonzero(y_train==0)[0]
spam_train = np.nonzero(y_train==1)[0]
print "Number of nonspam data in Training set ",len(nonspam_train)
print "Number of spam data in Training set ",len(spam_train)
#count of spam and non spam data in test set
nonspam_test = np.nonzero(y_test==0)[0]
spam_test = np.nonzero(y_test==1)[0]
print "Number of nonspam data in Test set ",len(nonspam_test)
print "Number of spam data in Test set ",len(spam_test)
#computing prior probability
p_spam= float(len(spam_train))/(float(len(spam_train))+ float(len(nonspam_train)))
p_nonspam= float(len(nonspam_train))/(float(len(spam_train))+ float(len(nonspam_train)))
print "Prior probability of spam data", p_spam
print "Prior probability of nonspam data", p_nonspam
mean_0=np.mean(x_train[np.nonzero(y_train==0)[0],:],axis=0)#Mean of each attribute with 0 as output
std_0=np.std(x_train[np.nonzero(y_train==0)[0],:],axis=0)#standard deviation of each attribute with 0 as output
std_0=np.add(std_0, 0.0001) #Adding an epsilon value to avoid standard deviation equal to zero
mean_1=np.mean(x_train[np.nonzero(y_train==1)[0],:],axis=0)#Mean of each attribute with 1 as output
std_1=np.std(x_train[np.nonzero(y_train==1)[0],:],axis=0)#standard deviation of each attribute with 1 as output
std_1=np.add(std_1, 0.0001)
#converting array to a list
mean1= mean_1.tolist()
mean0= mean_0.tolist()
#Array of both the means
mean=np.array([mean0,mean1]).T
#converting array to a list
std1= std_1.tolist()
std0= std_0.tolist()
#Array of both the standard deviations
std=np.array([std0,std1]).T
#Computing log of the prior probabilities
log_class=np.log([p_nonspam,p_spam])
y_pred=[] #Prediction list
#Gaussian naive bayed
for i in range(0, x_test.shape[0]):
	index=-1*(np.divide(np.power(np.subtract(x_test[i,:].reshape(x_test.shape[1],1),mean), 2),2*np.power(std, 2)))
	exponent=np.exp(index)
	if (np.any(np.nonzero(exponent==0)[0])):
		np.place(exponent,exponent==0,0.1e-200) #To avoid log(0) error
	pdf = np.divide(exponent,math.sqrt(2*np.pi)*std) #Probability density function
	pred = np.argmax(log_class+np.sum(np.log(pdf), axis=0))#Predicting the output from the model
	y_pred.append(pred) #List of all the predictions
print "\nAccuracy"
print accuracy_score(y_pred,y_test) #Accuracy is calculated
print "\nPrecision"
print precision_score(y_test, y_pred) #Precision is calculated
print "\nRecall"
print recall_score(y_test, y_pred) #Recall is calculated
print "\nConfusion matrix"
print confusion_matrix(y_test,y_pred) #Confusuin matrix is drawn


#Part 2- Classification with Logistic Regression
print "Logistic Regression"
lr=LogisticRegression() #Logistic Regression function from scikit learn library
lr.fit(x_train,y_train) #model is fitted
y_pred=lr.predict(x_test) #Predicting the test data based on the model
print "Default parameter values"
parameters=lr.get_params() #Parameter values are found and printed
print "Parameters: ",parameters
print "\nAccuracy"
print accuracy_score(y_pred,y_test)#Accuracy is calculated
print "\nPrecision"
print precision_score(y_test, y_pred) #Precision is calculated
print "\nRecall"
print recall_score(y_test, y_pred) #Recall is calculated
print "Confusion matrix"
print confusion_matrix(y_test,y_pred)






