import keras
from dataprocess import get_data_one
from keras.models import Sequential,Model,Input
from keras.callbacks import TensorBoard
from keras.layers import Dense
from keras.metrics import *
from keras.losses import mse

def getmodel(inunit,outunit):
    x=Input(shape=(10,))
    o=Dense(outunit,activation=None)(x)
    model=Model(x,o,name='model')
    return model

model=getmodel(10,1)

ten=TensorBoard(log_dir='./xxxx')

model.compile(optimizer='sgd',loss=mse)

x,y=get_data_one('/home/cooper/PycharmProjects/sxyl/Assignments/diabetes.txt')
model.fit(x,y,
          batch_size=32,
          epochs=1000,
          callbacks=[ten],
          validation_split=0.2)