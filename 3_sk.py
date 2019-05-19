from sklearn.linear_model import LogisticRegressionCV,LogisticRegression,SGDClassifier
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split
from dataprocess import get_data_two,get_data_three
from sklearn.metrics import precision_recall_curve,precision_score,recall_score,f1_score
import numpy as np
import keras
from dataprocess import get_data_one
from keras.models import Sequential,Model,Input
from keras.callbacks import TensorBoard,ModelCheckpoint
from keras.layers import Dense
from keras.metrics import *
from keras.losses import mse,categorical_crossentropy

X,Y=get_data_three('/home/cooper/PycharmProjects/sxyl/Assignments/iris3.txt')
stand=StandardScaler()
X=stand.fit_transform(X)
# onehot=OneHotEncoder()
# Y=onehot.fit_transform(Y)
print(X.shape,Y.shape)
# Y=np.asarray(Y,np.int)
YY=np.zeros([150,3],np.int64)
for i in range(Y.shape[0]):
    YY[i,Y[i]]=1
print(X.shape,YY.shape)
x_train,x_test,y_train,y_test=train_test_split(X,YY,test_size=0.2,random_state=1)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)



def getmodel(inunit,outunit):
    x=Input(shape=(4,))
    o=Dense(outunit,activation='softmax')(x)
    model=Model(x,o,name='model')
    return model

model=getmodel(4,3)

ten=TensorBoard()

model.compile(optimizer='sgd',loss=categorical_crossentropy,metrics=[categorical_accuracy])

# x,y=get_data_one('/home/cooper/PycharmProjects/sxyl/Assignments/diabetes.txt')
model.fit(x_train,y_train,
          batch_size=32,
          epochs=10000,
          callbacks=[ten],validation_data=[x_test,y_test])
model.save_weights('3_weight_model.hdf5')
# model.evaluate(x_test,y_test)

