# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 00:00:53 2018

@author: wushaowu
"""
from nn_model import nn_model1,nn_model2
from keras.models import load_model
import numpy as np  
import pandas as pd  
import os
os.environ['KERAS_BACKEND']='tensorflow'#'theano'
from keras.utils.np_utils import to_categorical
from collections import Counter
seed=7
np.random.seed(seed)

if __name__ == '__main__':
    """程序入口"""
    #读入特征数据
    train_y=pd.read_csv('label.csv')
    nn_stack_5zhe_train_feat= pd.read_csv('nn_stack_5zhe_train_feat.csv')
    nn_stack_5zhe_test_feat= pd.read_csv('nn_stack_5zhe_test_feat.csv')
    nn_stack_10zhe_train_feat= pd.read_csv('nn_stack_10zhe_train_feat.csv')
    nn_stack_10zhe_test_feat= pd.read_csv('nn_stack_10zhe_test_feat.csv')    
        
    #特征融合
    new_tr=pd.concat([nn_stack_10zhe_train_feat,nn_stack_5zhe_train_feat],axis=1)
    new_te=pd.concat([nn_stack_10zhe_test_feat,nn_stack_5zhe_test_feat],axis=1)
    
    if not os.path.exists('model1.h5'):
        #调用第一个神经网络模型
        print('第一个模型运行中...')
        model1=nn_model1(new_tr,to_categorical(train_y))
        model1.summary()
        pre1=model1.predict(new_te) ##预测
        print(Counter(np.argmax(pre1,axis=1)))
        model1.save('model1.h5') ##保存模型，下次可以直接调用
        np.savetxt("dsjyycxds_preliminary1.txt",np.argmax(pre1,axis=1).astype(int),fmt="%d") #保存第一个模型的结果
        print('第一个模型运行结束！！！')
        
        #调用第二个神经网络模型
        print('第二个模型运行中...')
        model2=nn_model2(new_tr,to_categorical(train_y))
        model2.summary()
        pre2=model2.predict(new_te) ##预测
        print(Counter(np.argmax(pre2,axis=1)))
        model2.save('model2.h5') ##保存模型，下次可以直接调用
        np.savetxt("dsjyycxds_preliminary2.txt",np.argmax(pre2,axis=1).astype(int),fmt="%d") #保存第二个模型的结果
        print('第二个模型运行结束！！！')
    else:
        #加载之前保存的模型
        model1=load_model('model1.h5')
        model2=load_model('model2.h5')
        #预测
        print('第一个模型运行中...')
        model1.summary()
        pre1=model1.predict(new_te)
        print('第一个模型的结果统计',Counter(np.argmax(pre1,axis=1)))
        np.savetxt("dsjyycxds_preliminary1.txt",np.argmax(pre1,axis=1).astype(int),fmt="%d")
        
        print('第二个模型运行中...')
        model2.summary()
        pre2=model2.predict(new_te)
        print('第二个模型的结果统计',Counter(np.argmax(pre2,axis=1)))
        np.savetxt("dsjyycxds_preliminary2.txt",np.argmax(pre2,axis=1).astype(int),fmt="%d")
    
    ##对上面的两个结果进行融合：
    y1=np.argmax(pre1,axis=1)
    y2=np.argmax(pre2,axis=1)
    res=[]
    for i in range(100000):
        if y1[i] in [1,7]:
            res.append(y1[i])
        else:
            res.append(y2[i])
    print(Counter(res))
    np.savetxt("dsjyycxds_preliminary.txt",np.array(res).astype(int),fmt="%d") #最终结果，提交
    