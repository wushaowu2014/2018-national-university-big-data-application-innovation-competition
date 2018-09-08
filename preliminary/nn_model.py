# -*- coding: utf-8 -*-
"""
Created on Sun May 20 17:24:03 2018

@author: shaowu
"""
import numpy as np 
import os
os.environ['KERAS_BACKEND']='tensorflow'#'theano'
from keras.layers import Dense,Dropout,Convolution1D,Flatten,Conv1D, BatchNormalization,PReLU
from keras.layers import Input, Embedding,concatenate,add,average,multiply,maximum
from keras.models import Model
from keras.optimizers import Adam
seed=7
np.random.seed(seed)

def bn_prelu(x):
    """定义标准化函数"""
    x = BatchNormalization()(x)
    x = PReLU()(x)
    return x

def nn_model1(train_x,train_y):
    """建立第一个三层的神经网络，不包括softmax层，下同"""
    inputs=Input(shape=(train_x.shape[1],))

    x1 = Dense(5, activation='tanh')(inputs)
    x2 = Dense(5, activation='relu')(inputs)
    x=multiply([x1,x2])
    x = Dense(20, activation='linear')(x)
    
    predictions = Dense(10, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(train_x, train_y,epochs=2, batch_size=200, validation_split=0.0,)
    return model

def nn_model2(train_x,train_y):
    """建立第二个五层的神经网络"""
    inputs=Input(shape=(train_x.shape[1],))
    
    x1 = Dense(500, activation='tanh')(inputs)
    x1 = bn_prelu(x1)
    x2 = Dense(500, activation='relu')(inputs)
    x2 = bn_prelu(x2)
    x=maximum([x1,x2])
    x = Dense(50, activation='sigmoid')(x)
    x = bn_prelu(x)
    x = Dense(50, activation='relu')(x)
    x = Dense(50, activation='linear')(x)
    
    predictions = Dense(10, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=Adam(lr=0.001, epsilon=1e-09, decay=0.0),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    model.fit(train_x, train_y,epochs=2, batch_size=200, validation_split=0.0)
    return model


