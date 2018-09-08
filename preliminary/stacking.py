# -*- coding: utf-8 -*-
"""
Created on Sun May 20 17:24:03 2018

@author: shaowu
"""
import numpy as np  
import pandas as pd  
import itertools  
import os
os.environ['KERAS_BACKEND']='tensorflow'#'theano'
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dense,Dropout,Convolution1D,Flatten,Conv1D, BatchNormalization,PReLU
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from scipy import sparse
from sklearn.utils import shuffle
from sklearn.cross_validation import StratifiedKFold
seed=7
np.random.seed(seed)

def lab_one_coder(df_train,df_test):
    """one-hot编码"""
    enc=OneHotEncoder()
    lb=LabelEncoder()
    feats=df_train.columns
    for k,feat in enumerate(feats):
        tmp=lb.fit_transform((list(df_train[feat])+list(df_test[feat])))
        enc.fit(tmp.reshape(-1,1))
        x_train=enc.transform(lb.transform(df_train[feat]).reshape(-1, 1))
      #  x_val=enc.transform(lb.transform(df_val[feat]).reshape(-1, 1))
        x_test=enc.transform(lb.transform(df_test[feat]).reshape(-1, 1))
        if k==0:
            X_train,X_test=x_train,x_test
        else:
            X_train,X_test=sparse.hstack((X_train, x_train)),sparse.hstack((X_test, x_test))
    return X_train,X_test

def gather_data(data):
    """对除0和3类之外的类做全排列"""
    newdata=pd.DataFrame()
    col_dict={'ziduan1':'num1','ziduan2':'num2','ziduan3':'num3','ziduan4':'num4','ziduan5':'num5'}
    col_list=list(itertools.permutations(list(col_dict.keys()),5))#[:60]
    for i in col_list:
        col=[]
        for j in i:
            col=col+[j,col_dict.get(j)]
        m=data[col+['label']]
        m.columns=data.columns
        newdata=pd.concat([newdata,m])
    return newdata

def gather_data03(data):
    """对0和3类做全排列"""
    newdata=pd.DataFrame()
    col_dict={'ziduan1':'num1','ziduan2':'num2','ziduan3':'num3','ziduan4':'num4','ziduan5':'num5'}
    col_list=list(itertools.permutations(list(col_dict.keys()),5))
    for i in col_list:
        col=[]
        for j in i:
            col=col+[j,col_dict.get(j)]
        m=data[col+['label']]
        m.columns=data.columns
        newdata=pd.concat([newdata,m])
    return newdata

def bujiehe_model(train_x,train_y):
    """建立三层的神经网络，不包括softmax层"""
    model = Sequential()
    model.add(Dense(1000, input_shape=(train_x.shape[1],)))
    model.add(Activation('tanh'))  

    model.add(Dense(500))  
    model.add(Activation('tanh'))  

    model.add(Dense(50))  
    model.add(Activation('linear'))
  

    model.add(Dense(10)) #这里需要和输出的维度一致  
    model.add(Activation('softmax'))  

    #三种优化器，任选一种：SGD、Adam 和 rmsprop
    model.compile(optimizer=#sgd,
              Adam(lr=0.001, epsilon=1e-07, decay=0.0),
              loss='categorical_crossentropy',  
              metrics=['accuracy'])  
  
    epochs = 5 #迭代次数
    early_stopping = EarlyStopping(monitor='loss', patience=2)
    model.fit(train_x, train_y, epochs=epochs, batch_size=800, validation_split=0.0,
          callbacks=[early_stopping])
    return model

def stacking(train_x,train_y,test):
    """做5折和10折stacking特征融合"""
    for n_folds in [5,10]:
        print(str(n_folds)+'_folds_stacking')
        stack_train = np.zeros((len(train_y), 10))
        stack_test = np.zeros((test.shape[0], 10))
        score_va = 0
        train_y=train_y.astype(int)
        for i, (tr, va) in enumerate(StratifiedKFold(train_y, n_folds=n_folds, random_state=2018)):
            print('stack:%d/%d' % ((i + 1), n_folds))
            model=bujiehe_model(train_x.iloc[tr],to_categorical(train_y[tr]))
            score_va = model.predict(train_x.iloc[va])
            score_te = model.predict(test)
            stack_train[va] += score_va
            stack_test += score_te
        stack_test /= n_folds
        df_stack_train = pd.DataFrame()
        df_stack_test = pd.DataFrame()
        for i in range(stack_test.shape[1]):
            df_stack_train['nnmodel_'+str(n_folds)+'_classfiy_{}'.format(i)] = stack_train[:, i]
            df_stack_test['nnmodel_'+str(n_folds)+'_classfiy_{}'.format(i)] = stack_test[:, i]
        df_stack_train.to_csv('nn_stack_'+str(n_folds)+'zhe_train_feat.csv', index=None, encoding='utf8')
        df_stack_test.to_csv('nn_stack_'+str(n_folds)+'zhe_test_feat.csv', index=None, encoding='utf8')
        print(str(n_folds)+'_folds_stacking特征已保存\n')

def processing():
    """数据的预处理工作"""
    #读入数据：
    train = pd.read_csv('training.csv',header=None) #训练数据路径
    test1 = pd.read_csv('preliminary-testing.csv',header=None) #测试数据路
    
    #添加字段名，方便后面处理
    train.columns=['ziduan1','num1','ziduan2','num2','ziduan3','num3','ziduan4','num4','ziduan5','num5','label']
    test1.columns=['ziduan1','num1','ziduan2','num2','ziduan3','num3','ziduan4','num4','ziduan5','num5']

    ###把除0和3类外的其他类进行全排列
    train_x=train
    train=train.ix[train['label']!=0]
    train=train.ix[train['label']!=3]
    train=gather_data(train)
    train=pd.concat([train,train_x.ix[train_x['label']==0]])
    train1=pd.concat([train,train_x.ix[train_x['label']==3]])

    ##对0类和3类进行全排列：
    train03=train_x.ix[train_x['label']==0]
    train03=pd.concat([train03,train_x.ix[train_x['label']==3]])
    train03=gather_data03(train03).reset_index(drop=True)

    ##结合全排列后的数据，并打乱顺序
    train=pd.concat([train1,train03])
    train=shuffle(train)
    train=train.drop_duplicates().reset_index(drop=True)
    
    train_y=train['label']
    train[['label']].to_csv('label.csv',index=None)#保存标签，以备往后方便使用
    train_x=train.drop(['label'],axis=1)
    
    ##进行one-hot编码
    train_x,test=lab_one_coder(train_x,test1)
    train_x=train_x.toarray()
    test=test.toarray()
    train_x=pd.DataFrame(train_x)
    test=pd.DataFrame(test)
    
    ###进行5折，10折stacking特征融合
    stacking(train_x,train_y,test)
    
if __name__ == '__main__':
    """程序入口"""
    processing()