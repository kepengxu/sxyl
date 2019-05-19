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


def get_data_two(path):
    X=[]
    Y=[]
    with open(path) as f:
        for line in f:
            words=line.split()
            if not len(words)==5:
                break
            x=[float(i) for i in words[:-1]]
            y=float(words[4])
            # print(x,y)

            X.append(x)
            Y.append(y)
    X=np.array(X,np.float32)
    Y=np.array(Y,np.int)
    Y=Y[:,np.newaxis]
    print(X.shape,Y.shape)
    return X,Y

def get_data_three(path):
    X=[]
    Y=[]
    with open(path) as f:
        for line in f:
            words=line.split()
            if not len(words)==5:
                break
            x=[float(i) for i in words[:-1]]
            y=float(words[-1])
            X.append(x)
            Y.append(y)
    X=np.array(X,np.float32)
    Y=np.array(Y,np.int)
    Y=Y[:,np.newaxis]

    print(X.shape,Y.shape)
    return X,Y





if __name__=='__main__':
#    x,y=get_data_one('/home/cooper/PycharmProjects/sxyl/Assignments/diabetes.txt')
     x,y=get_data_three('/home/cooper/PycharmProjects/sxyl/Assignments/iris3.txt')