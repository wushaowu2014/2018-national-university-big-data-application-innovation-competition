# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 00:00:53 2018

@author: wushaowu
"""
from utils import predict_pre_sample,model1,gather_data,feat_onehot
from keras.models import load_model
import numpy as np  
import pandas as pd
from scipy import stats
import os
os.environ['KERAS_BACKEND']='tensorflow'#'theano'
from keras.utils.np_utils import to_categorical
from collections import Counter
seed=7
np.random.seed(seed)
def pre1(x):
    if np.argsort(x)[-1]==1:
        if np.max(x)>0.99995:
            return np.argsort(x)[-1]
        elif np.argsort(x)[-2]==5:
            return np.argsort(x)[-3]
        else:
            return np.argsort(x)[-2]
    elif np.argsort(x)[-1]==6:
        if np.max(x)>0.98:
            return np.argsort(x)[-1]
        else:
            return 5
    else:
        return np.argsort(x)[-1]

if __name__ == '__main__':
    """程序入口"""
    #读入特征数据
    train=pd.read_csv('train.csv')
    test=pd.read_csv('test.csv')
    label=train['label']
    
    nnmodel1_5folds_train_feat= pd.read_csv('nnmodel1_5folds_train_feat.csv')
    nnmodel1_5folds_test_feat= pd.read_csv('nnmodel1_5folds_test_feat.csv')
    nnmodel2_5folds_train_feat= pd.read_csv('nnmodel2_5folds_train_feat.csv')
    nnmodel2_5folds_test_feat= pd.read_csv('nnmodel2_5folds_test_feat.csv')
    nnmodel3_5folds_train_feat= pd.read_csv('nnmodel3_5folds_train_feat.csv')
    nnmodel3_5folds_test_feat= pd.read_csv('nnmodel3_5folds_test_feat.csv')

    #特征结合
    new_tr=pd.concat([nnmodel1_5folds_train_feat,\
                      nnmodel2_5folds_train_feat,\
                      nnmodel3_5folds_train_feat,\
                      pd.DataFrame(feat_onehot(train.drop(['label'],axis=1),test.columns))
                      ],axis=1)
    new_te=pd.concat([nnmodel1_5folds_test_feat,\
                      nnmodel2_5folds_test_feat,\
                      nnmodel3_5folds_test_feat,\
                      pd.DataFrame(feat_onehot(test,test.columns))
                      ],axis=1)
    
    ###################################预测#####################################
    if not os.path.exists('pre_model.h5'):
        model=model1(new_tr,to_categorical(label)) #调用神经网络预测模型
        model1.summary()
        results=model.predict(new_te) ##预测
        print(Counter(np.argmax(results,axis=1)))
        model.save('pre_model.h5') ##保存模型，下次可以直接调用
    else:
        model=load_model('pre_model.h5') #加载之前保存的预测模型
        #model.summary()
        results=model.predict(new_te) #预测
        print('模型的结果统计',Counter(np.argmax(results,axis=1)))
    
    ####################################后处理##################################
    #提前载入stacking模型：
    model1_0=load_model('stack_model1/5_folds_stack_model0.h5')
    model1_1=load_model('stack_model1/5_folds_stack_model1.h5')
    model1_2=load_model('stack_model1/5_folds_stack_model2.h5')
    model1_3=load_model('stack_model1/5_folds_stack_model3.h5')
    model1_4=load_model('stack_model1/5_folds_stack_model4.h5')
    
    model2_0=load_model('stack_model2/5_folds_stack_model0.h5')
    model2_1=load_model('stack_model2/5_folds_stack_model1.h5')
    model2_2=load_model('stack_model2/5_folds_stack_model2.h5')
    model2_3=load_model('stack_model2/5_folds_stack_model3.h5')
    model2_4=load_model('stack_model2/5_folds_stack_model4.h5')

    model3_0=load_model('stack_model3/5_folds_stack_model0.h5')
    model3_1=load_model('stack_model3/5_folds_stack_model1.h5')
    model3_2=load_model('stack_model3/5_folds_stack_model2.h5')
    model3_3=load_model('stack_model3/5_folds_stack_model3.h5')
    model3_4=load_model('stack_model3/5_folds_stack_model4.h5')
    
    print('后处理工作...')
    final_results=[]
    for i in range(len(results)):
        if ((np.max(results[i])<0.99995) and (np.argsort(results[i])[-1] in [0,2])) or\
        ((np.max(results[i])<0.72) and np.argsort(results[i])[-1]==7):
            initial_sample=gather_data(test[i:i+1]).reset_index(drop=True) #全排列
            #结合该样本的特征：
            new_sample=np.hstack((
                   0.2*(model1_0.predict(initial_sample)+\
                       model1_1.predict(initial_sample)+\
                       model1_2.predict(initial_sample)+\
                       model1_3.predict(initial_sample)+\
                       model1_4.predict(initial_sample)),\
                   0.2*(model2_0.predict(initial_sample)+\
                       model2_1.predict(initial_sample)+\
                       model2_2.predict(initial_sample)+\
                       model2_3.predict(initial_sample)+\
                       model2_4.predict(initial_sample)),\
                   0.2*(model3_0.predict(initial_sample)+\
                       model3_1.predict(initial_sample)+\
                       model3_2.predict(initial_sample)+\
                       model3_3.predict(initial_sample)+\
                       model3_4.predict(initial_sample)),\
                       feat_onehot(initial_sample,test.columns)
                      ))
            res=model.predict(new_sample) #预测出120个结果
            
            if np.max(res.max(axis=1))>0.99999 and np.argsort(results[i])[-1]==0:
                x=stats.mode([
                np.argmax(res[[np.argsort(res.max(axis=1))[-1]]]),\
                np.argmax(res[[np.argsort(res.max(axis=1))[-2]]]),\
                np.argmax(res[[np.argsort(res.max(axis=1))[-3]]]),\
                np.argmax(res[[np.argsort(res.max(axis=1))[-4]]]),\
                np.argmax(res[[np.argsort(res.max(axis=1))[-5]]]),\
                ])[0][0]

                xx=stats.mode([np.argmax(res[np.argmax(res.max(axis=1))]),\
                              sorted(Counter(np.argmax(res,axis=1)).items(), key=lambda d: d[1])[-1][0],\
                              x])[0][0]
 
                final_results.append(xx)
            elif np.max(res.max(axis=1))>0.99998:
                final_results.append(np.argmax(res[np.argmax(res.max(axis=1))])) #取最大概率的类别
            elif np.max(res.max(axis=1))>0.99995 and np.argsort(results[i])[-1]==2:
                final_results.append(res[1].argsort()[-1]) 
            else:
                final_results.append(sorted(Counter(np.argmax(res,axis=1)).items(), key=lambda d: d[1])[-1][0])
            
        else:
            final_results.append(pre1(results[i]))
    
    #结果保存:   
    np.savetxt("dsjyycxds_semifinal.txt",np.array(final_results).astype(int),fmt="%d")