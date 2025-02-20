#Author: Prajwal Y P
#CreatedDate: 2025-02-20
#Purpose: To displaying the basic working of liearn regression model/function working

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Univariate Linear Regression Model | Calculate the estimated price of house
def UnivariateLinearRegressionModel(w,x,b):
    return w*x+b

#Training data set
xTrain=np.array([1.0,2.0])
yTrain=np.array([300,500])

gs = gridspec.GridSpec(4, 2)

#draw a Training set graph
ax1=plt.subplot(gs[0,:2])
ax1.scatter(x=xTrain,y=yTrain,marker='x',c='red')
ax1.set_title('Training Set')
ax1.set_xlabel('Size of the house')
ax1.set_ylabel('price of the house')


#Model parameters guess -1
w=100
b=100

ax2=plt.subplot(gs[1,0])
ax2.plot(xTrain,[UnivariateLinearRegressionModel(w,xTrain[0],b),UnivariateLinearRegressionModel(w,xTrain[1],b)])
ax2.scatter(x=xTrain,y=yTrain,marker='x',c='red')
ax2.set_title('Model parameters guess -1')
ax2.set_xlabel('Size of the house')
ax2.set_ylabel('price of the house')


for i in range(len(xTrain)):
    print(f'{xTrain[i]} sq.ft. size of house costs {UnivariateLinearRegressionModel(w,xTrain[i],b)}')

#Model parameters guess -2
w=200
b=100

ax3=plt.subplot(gs[1,1])
ax3.plot(xTrain,[UnivariateLinearRegressionModel(w,xTrain[0],b),UnivariateLinearRegressionModel(w,xTrain[1],b)])
ax3.scatter(x=xTrain,y=yTrain,marker='x',c='red')
ax3.set_title('Model parameters guess -2')
ax3.set_xlabel('Size of the house')
ax3.set_ylabel('price of the house')


#Guess the proce for new house

newSize=1.3
newPrice=UnivariateLinearRegressionModel(w,newSize,b)

ax4=plt.subplot(gs[2:4,:2])
ax4.plot(xTrain,[UnivariateLinearRegressionModel(w,xTrain[0],b),UnivariateLinearRegressionModel(w,xTrain[1],b)],label='func')
ax4.plot([0,newSize],[newPrice,newPrice],c='orange',linestyle='--',marker='o',label='predicted val')
ax4.plot([newSize,newSize],[newPrice,0],c='orange',linestyle='--',marker='o')
ax4.scatter(x=xTrain,y=yTrain,marker='x',c='red',label='TrainingSet')
ax4.set_title('Working of Linear Regression')
ax4.set_xlabel('Size of the house')
ax4.set_ylabel('price of the house')
ax4.legend()

plt.box(1)
plt.tight_layout()
plt.show()
