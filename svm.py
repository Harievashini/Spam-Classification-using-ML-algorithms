from sklearn import metrics
from sklearn.metrics import *
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC 
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plot
import numpy as np 
import random


#loading the dataset 
data = np.loadtxt('spambase.data.csv', delimiter=',')
#splitting into attributes and labels 
x = data[:,:57] #Attribute set
y=data[:,57] #Label set
 #splitting into train and test data set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.50) 
mean_x=np.mean(x_train, axis = 0)#Mean of each attribute
std_x=x_train.std(0)#standard deviation of each attribute
#creating copies to perform scaling
mean_X = np.tile(mean_x, (2300, 1))   
std_X = np.tile(std_x, (2300, 1))   
#Data preprocessing 
X_train=(x_train- mean_X)/ std_X #Preprocessed attributes or features of training set
#Using the mean and standard deviation of training set to scale test set
mean_X=np.tile(mean_x, (2301, 1)) 
std_X = np.tile(std_x, (2301, 1)) 
X_test=(x_test- mean_X)/ std_X #Preprocessed attributes or features of test set


#Experiment 1

print "Experiment 1"
svclassifier = SVC(kernel='linear')  #Svm uses linear kernel
svclassifier.fit(X_train, y_train)  #model is fitted based on training set
y_pred = svclassifier.predict(X_test) #Predicting the test data based on the model
Accuracy=accuracy_score(y_pred,y_test) #Accuracy of the test dataset is calculated
print "\nAccuracy: "
print Accuracy
print "\nPrecision"
print precision_score(y_test, y_pred) #Precision calculation
print "\nRecall"
print recall_score(y_test, y_pred) #Recall calculation
print(classification_report(y_test, y_pred))
#roc curve is generated
scores=svclassifier.decision_function(X_test)
fpr,tpr, thresholds=roc_curve(y_test, scores) #false positive rate and true positive rate is calculated
auc=auc(fpr, tpr) #area under curve

#Roc curve graph taking fpr and tpr in x axis and y axis
plot.figure(1)
plot.plot(fpr, tpr, color='orange', lw=1, label='ROC curve (auc = %f)' % auc)
plot.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plot.xlabel('False Positive Rate')
plot.ylabel('True Positive Rate')
plot.title('Receiver operating characteristic')
plot.legend(loc="lower right")


#Experiment 2

print "Experiment 2"
# coefficients are determined- weight of each attribute
coefs=svclassifier.coef_
print "Coefficients: "
print coefs
coef=np.argsort(coefs) #sorting the features based on their weights
m_features=[] #List of m values
accuracies=[] #list of accuracies
n=56 #index
print coef[0][n], "and"
n-=1
X_train_max=np.array(X_train[:,coef[0][n]],ndmin=2).T #The training set taking features with high weight coefficient
X_test_max=np.array(X_test[:,coef[0][n]],ndmin=2).T #The test set taking features with high weight coefficients
# considering m features with high weights , where m ranges from 2 to 57
for m in range(2,58):
    print coef[0][n],"th feature is added"
    X_train_max=np.insert(X_train_max,0,X_train[:,coef[0][n]],axis=1)#The training set taking features with high weight coefficient
    X_test_max=np.insert(X_test_max,0,X_test[:,coef[0][n]],axis=1)#The test set taking features with high weight coefficient
    n-=1
    #model is fitted based on training set
    svclassifier.fit(X_train_max,y_train)
    #Run test data with the model
    y_pred=svclassifier.predict(X_test_max)
    # Determining accuracy from prediction and actual output
    acc=accuracy_score(y_pred,y_test)
    m_features.append(m) #Appending the m values
    accuracies.append(acc) #Appending the accuracies
    print "m=",m, "Accuracy= ",acc

# Graph of m vs accuracies where the features are chosen based on its weights	
plot.figure(2)
plot.title("Feature Selection with Linear SVM")
plot.plot(m_features,accuracies)
plot.ylim(0.60,0.95)
plot.xlabel("m")
plot.ylabel("Accuracy")


#Experiment 3

print "Experiment 3"
#List of m values and List of accuracies- reinitializing
m_features=[]
accuracies=[]
#Considering m features, where m ranges from 2 to 57
for m in range(2,58):
    feature_index=random.sample(range(0,57),  m) # m random number are generated within the range 0 to 57
    print feature_index,"feature "
    X_train_max=X_train[:,feature_index] #Training set of m features taken at random
    X_test_max=X_test[:,feature_index] #Test set of m features taken at random
    #model is fitted based on training set
    svclassifier.fit(X_train_max,y_train)
    #Run test data with the model
    y_pred=svclassifier.predict(X_test_max)
    # Determining accuracy from prediction and actual output
    acc=accuracy_score(y_pred,y_test)
    m_features.append(m)#Appending the m values
    accuracies.append(acc)#Appending the accuracies
    print "m=",m, "Accuracy= ",acc

# Graph of m vs accuracies where the features are chosen at random	
plot.figure(3)
plot.title("Random feature selection")
plot.plot(m_features,accuracies)
plot.xlabel("m")
plot.ylabel("Accuracy")

plot.show()
                                    

