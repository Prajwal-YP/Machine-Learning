"""
Module: Linear Regression Implementation
Author: Prajwal Y P
Created: 2025-02-20
Purpose: To demonstrate the basic working of a linear regression model.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridSpec

class LinearRegression:
    """
    A simple linear regression model.
    
    Attributes:
        w (float): The weight of the regression line.
        b (float): The bias of the regression line.
    """

    def __init__(self,w:float,b:float):
        """
        Initializes the LinearRegression model with weight and bias.
        
        Parameters:
            w (float): The weight of the regression line.
            b (float): The bias of the regression line.
        """

        self.w=w
        self.b=b
    
    def set_TrainingData(self,xTrain:np.array,yTrain:np.array):
        """
        Sets the training data for the linear regression model.
        
        Parameters:
            xTrain (np.array): The input features.
            yTrain (np.array): The target values.
        """

        self.x=xTrain
        self.y=yTrain
    
    def predict(self,newinput:float):
        """
        Predicts the output for a given input using the linear regression model.
        
        Parameters:
            newinput (float): The input value.
        
        Returns:
            float: The predicted output value.
        """   
        
        output=self.w*newinput+self.b
        return output
    
    def buildGraph(self,newIp:float,newOp:float):
        """
        Builds and displays graphs for the training data, the regression line, and the predicted value.
        
        Parameters:
            newIp (float): The new input value.
            newOp (float): The corresponding predicted output value.
        """
        
        gs=gridSpec.GridSpec(3,1,wspace=0,hspace=1)

        #Plot all the data points of training set
        ax1=plt.subplot(gs[0,0])
        ax1.set_xlabel('Size of the house')
        ax1.set_ylabel('Price of the house')
        ax1.scatter(self.x,self.y,c='red',marker='x')
        ax1.set_title('Training set')

        #Plot all training data-set along with the linear function
        ax2=plt.subplot(gs[1,0])
        ax2.set_xlabel('Size of the house')
        ax2.set_ylabel('Price of the house')
        ax2.scatter(self.x,self.y,c='red',marker='o')
        ax2.plot(self.x,self.w*self.x+self.b,c='blue')
        ax2.set_title('Guessed Set/Model')

        #PLot all training data-set, linear function and the newly predicted value
        ax3=plt.subplot(gs[2,0])
        ax3.set_xlabel('Size of the house')
        ax3.set_ylabel('Price of the house')
        ax3.scatter(self.x,self.y,c='red',marker='o')
        ax3.plot(self.x,self.w*self.x+self.b,c='blue',label='linear function')
        ax3.plot([0,newIp],[newOp,newOp],c='orange',marker='s',linestyle='--',label='predicted value')
        ax3.plot([newIp,newIp],[0,newOp],c='orange',marker='s',linestyle='--')
        ax3.set_title(f'Predicted (Size:{newIp}, Price:{newOp})')
        
        plt.legend()
        plt.show()


def get_parameters():
    """
        Take input for weight and bias parameters and validates that they are of numeric data type

        return: (float,float)  returns validated parameter values
    """
    gotW=False
    gotB=False

    while True:
        try:
            if not gotW:
                w=float(input('\tw:\t'))
                gotW=True
        except:
            print('Enter numeric value for w')
            continue

        try:
            if not gotB:
                b=float(input('\tb:\t'))
                gotB=True
        except:
            print('Enter numeric value for b')
            continue

        if gotW and gotB:
            return (w,b)


#Create linear regression with guess 1 parameters
print('Enter guess-1 w and b parameters:')
w,b=get_parameters()
lr1=LinearRegression(w=w,b=b)
lr1.set_TrainingData(xTrain=np.array([1.0,2.0]),yTrain=np.array([300,500]))
lr1.buildGraph(1.3,lr1.predict(1.3))

#Create linear regression with guess 2 parameters
print('Enter guess-2 w and b parameters:')
w,b=get_parameters()
lr2=LinearRegression(w=w,b=b)
lr2.set_TrainingData(xTrain=np.array([1.0,2.0]),yTrain=np.array([300,500]))
lr2.buildGraph(1.3,lr2.predict(1.3))
