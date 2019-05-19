# encoding:utf-8
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV,LogisticRegression,SGDClassifier
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split
from dataprocess import get_data_two,get_data_two
from sklearn.metrics import precision_recall_curve,precision_score,recall_score,f1_score
import numpy as np
import keras
from dataprocess import get_data_one
from keras.models import Sequential,Model,Input
from keras.callbacks import TensorBoard,ModelCheckpoint
from keras.layers import Dense
from keras.metrics import *
from keras.losses import mse,categorical_crossentropy

stand=StandardScaler()
x,y=get_data_two('/home/cooper/PycharmProjects/sxyl/Assignments/iris2.txt')

x=stand.fit_transform(X=x)

print(x.shape)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


class GDA:
    def __init__(self,train_data,train_label):

        self.Train_Data = train_data
        self.Train_Label = train_label
        self.postive_num = 0                                                    # 正样本个数
        self.negetive_num = 0                                                   # 负样本个数
        postive_data = []                                                       # 正样本数组
        negetive_data = []                                                      # 负样本数组
        for (data,label) in zip(self.Train_Data,self.Train_Label):
            if label == 1:          # 正样本
                self.postive_num += 1
                postive_data.append(list(data))
            else:                   # 负样本
                self.negetive_num += 1
                negetive_data.append(list(data))
        # 计算正负样本的二项分布的概率
        row,col = np.shape(train_data)
        self.postive = self.postive_num*1.0/row                                 # 正样本的二项分布概率
        self.negetive = 1-self.postive                                          # 负样本的二项分布概率
        # 计算正负样本的高斯分布的均值向量
        postive_data = np.array(postive_data)
        negetive_data = np.array(negetive_data)
        postive_data_sum = np.sum(postive_data, 0)
        negetive_data_sum = np.sum(negetive_data, 0)
        self.mu_positive = postive_data_sum*1.0/self.postive_num                # 正样本的高斯分布的均值向量
        self.mu_negetive = negetive_data_sum*1.0/self.negetive_num              # 负样本的高斯分布的均值向量
        # 计算高斯分布的协方差矩阵
        positive_deta = postive_data-self.mu_positive
        negetive_deta = negetive_data-self.mu_negetive
        self.sigma = []
        for deta in positive_deta:
            deta = deta.reshape(1,col)
            ans = deta.T.dot(deta)
            self.sigma.append(ans)
        for deta in negetive_deta:
            deta = deta.reshape(1,col)
            ans = deta.T.dot(deta)
            self.sigma.append(ans)
        self.sigma = np.array(self.sigma)
        #print(np.shape(self.sigma))
        self.sigma = np.sum(self.sigma,0)
        self.sigma = self.sigma/row
        self.mu_positive = self.mu_positive.reshape(1,col)
        self.mu_negetive = self.mu_negetive.reshape(1,col)

    def Gaussian(self, x, mean, cov):
        """
        这是自定义的高斯分布概率密度函数
        :param x: 输入数据
        :param mean: 均值向量
        :param cov: 协方差矩阵
        :return: x的概率
        """
        dim = np.shape(cov)[0]
        # cov的行列式为零时的措施
        covdet = np.linalg.det(cov + np.eye(dim) * 0.001)
        covinv = np.linalg.inv(cov + np.eye(dim) * 0.001)
        xdiff = (x - mean).reshape((1, dim))
        # 概率密度
        prob = 1.0 / (np.power(np.power(2 * np.pi, dim) * np.abs(covdet), 0.5)) * \
               np.exp(-0.5 * xdiff.dot(covinv).dot(xdiff.T))[0][0]
        return prob

    def predict(self,test_data):
        predict_label = []
        for data in test_data:
            positive_pro = self.Gaussian(data,self.mu_positive,self.sigma)
            negetive_pro = self.Gaussian(data,self.mu_negetive,self.sigma)
            if positive_pro >= negetive_pro:
                predict_label.append(1)
            else:
                predict_label.append(0)
        return predict_label

def run_main():
    """
       这是主函数
    """
    # 导入乳腺癌数据
    breast_cancer = load_breast_cancer()
    data = np.array(breast_cancer.data)
    label = np.array(breast_cancer.target)
    print data.shape,label.shape

if __name__=='__main__':
    gda=GDA(x_train,y_train)
    y_pre=gda.predict(x_test)
    print('recall: ', recall_score(y_test, y_pre))
    print('precision', precision_score(y_test, y_pre))
    print('f1_score', f1_score(y_test, y_pre))