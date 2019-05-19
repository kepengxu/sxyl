from sklearn.linear_model import LogisticRegressionCV,LogisticRegression,SGDClassifier
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split
from dataprocess import get_data_two,get_data_three
from sklearn.metrics import precision_recall_curve,precision_score,recall_score,f1_score
import numpy as np
x,y=get_data_three('/home/cooper/PycharmProjects/sxyl/Assignments/iris3.txt')
stand=StandardScaler()
x=stand.fit_transform(x)
onehot=OneHotEncoder()
y=onehot.fit_transform(y)
print(x.shape,y.shape)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
model = LogisticRegressionCV(fit_intercept=True, Cs=np.logspace(-5, 1, 100),
                          multi_class='multinomial', penalty='l2', solver='lbfgs')

model.fit(x_train,y_train)

y_pre=model.predict(x_test)
print('recall: ',recall_score(y_test,y_pre))
print('precision',precision_score(y_test,y_pre))
print('f1_score',f1_score(y_test,y_pre))


