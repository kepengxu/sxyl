from sklearn.linear_model import LogisticRegressionCV,LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from dataprocess import get_data_two
from sklearn.metrics import precision_recall_curve,precision_score,recall_score,f1_score
import numpy as np

stand=StandardScaler()
x,y=get_data_two('/home/cooper/PycharmProjects/sxyl/Assignments/iris2.txt')

x=stand.fit_transform(X=x)

print(x.shape)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
model=LogisticRegressionCV(multi_class="ovr",fit_intercept=True,Cs=np.logspace(-2,2,20),cv=2,penalty="l2",solver="lbfgs",tol=0.01)
#
result=model.fit(x_train,y_train)
#
s=result.score(x_train,y_train)
print(s)
y_pre=model.predict(x_test)
print('recall: ',recall_score(y_test,y_pre))
print('precision',precision_score(y_test,y_pre))
print('f1_score',f1_score(y_test,y_pre))
