# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 08:45:14 2018

@author: wushaowu
"""
from utils import processing
if __name__ == '__main__':
    """程序入口"""
    train_path = 'training-final.csv' #训练数据路径
    test_path = 'Semifinal-testing-final.csv' #测试数据路径
    processing(train_path,test_path) #数据预处理
