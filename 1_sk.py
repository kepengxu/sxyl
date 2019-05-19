from dataprocess import get_data_one
x,y=get_data_one('/home/cooper/PycharmProjects/sxyl/Assignments/diabetes.txt')

print(x.shape,y.shape)
LEN=int(x.shape[0]*0.8)
x_train,y_train=x[:LEN],y[:LEN]
x_test,y_test=x[LEN:],y[LEN:]
from sklearn import linear_model
model=linear_model.LinearRegression(n_jobs=-1)

model.fit(x_train,y_train)
print(model.intercept_)  #截距
print(model.coef_)  #线性模型的系数

testPred=model.predict(x_test)
import numpy as np

error=testPred-y_test
print(error)



