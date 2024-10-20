import numpy as np
class LinearRegression:
    
    def __init__(self,lr = 0.01,n_iters=1000):
        self.lr = lr
        self.n_iters=n_iters
        self.weights = None
        self.bias = None 
        
    def fit(self,X,y):
        #to retrive number of features 
        n_samples, n_features = X.shape
        #to create the zero array and using number of features for creating zero array 
        self.weights=np.zeros(n_features)
        self.bias=0
        
        for _ in range(self.n_iters):
        #This is the simplified formula for prediction using the line equation : y=mX+c where X is [xi......]
            y_pred = np.dot(X,self.weights)+self.bias
            
            #calulating the derivaties the gradients : dw is for updating weights and db is for bias and formulas are as follow  dw = 1/N summation (1->n) 2xi(y_pred-y) and db = 1/N summation (1->n )2(y_pred -y)
            dw =(1/n_samples)*np.dot(X.T,(y_pred-y))
            db = (1/n_samples)*np.sum(y_pred-y)
            # updating the values where we subtract the weights - learning rate * derivaties 
            self.weights= self.weights-self.lr*dw
            self.bias = self.bias-self.lr*db
        
    def predict(self,X):
         y_pred = np.dot(X,self.weights)+self.bias
         return y_pred