# coding:utf-8
import numpy as np
def get_data_one(path):
    X=[]
    Y=[]
    with open(path) as f:
        for line in f:
            words=line.split()

            x=[float(i) for i in words[:-1]]
            y=float(words[-1])
            X.append(x)
            Y.append(y)
    X=np.array(X)
    Y=np.array(Y)
    Y=Y[:,np.newaxis]
    print(X.shape,Y.shape)
    return X,Y

if __name__=='__main__':
    x,y=get_data_one('/home/cooper/PycharmProjects/sxyl/Assignments/diabetes.txt')