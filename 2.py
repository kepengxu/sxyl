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

import keras
from dataprocess import get_data_one
from keras.models import Sequential,Model,Input
from keras.callbacks import TensorBoard
from keras.layers import Dense
from keras.metrics import *
from keras.losses import mse,binary_crossentropy

def getmodel(inunit,outunit):
    x=Input(shape=(inunit,))
    o=Dense(outunit,activation='sigmoid')(x)
    model=Model(x,o,name='model')
    return model

model=getmodel(4,1)

ten=TensorBoard(log_dir='./sigmoid',write_graph=True,write_grads=True,write_images=True)

model.compile(optimizer='sgd',loss=binary_crossentropy,metrics=[binary_accuracy])


model.fit(x,y,
          batch_size=32,
          epochs=1000,
          callbacks=[ten],
          validation_split=0.2)