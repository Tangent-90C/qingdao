import pandas as pd
import os
import numpy as np
import sys
sys.path.append('components')
from utils.feature_extraction import feature_extraction
from sklearn.feature_selection import SelectKBest
from utils.get_training_data import get_training_data
from utils.init_train_label import merge_label

import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

 
from utils.config import cfg
import json

#在utils文件夹下utils。py文件中修改训练集的路径

#特征提取,处理后的特征保存下来，如果没有值记做nan


if __name__ == '__main__':
    print('feature extraction')
    feature_extraction()

    #获取标签
    print('get label')
    merge_label()

    # 得到训练样本(含特征补全)
    print('training')
    samples,labels = get_training_data(cfg.train_feature_path)

    #sel = SelectKBest()
    #samples = sel.fit_transform(samples,labels)

    # 标准化
    # zscore_scaler = preprocessing.StandardScaler()
    # print(samples)
    # zscore_scaler.fit(samples)
    # samples_scaler = zscore_scaler.transform(samples)

    X_train,X_test,y_train,y_test = train_test_split(samples,labels,test_size=0.2,random_state=3407)

    print('训练集size：{}'.format(X_train.shape))
    print('训练集共：{}个NG数据。'.format(sum(y_train)))

    # 训练模型
    #model = xgb.XGBClassifier(max_depth=200,learning_rate=0.1,n_estimators=100,objective='binary:logistic')
    model = xgb.XGBRFClassifier()
    # model = xgb.XGBClassifier(max_depth=100,learning_rate=0.01,n_estimators=200,objective='binary:logitraw')

    model.fit(X_train,y_train)

    # 对测试集进行预测
    y_pred = model.predict(X_test)

    #计算准确率
    f1score = f1_score(y_test,y_pred)
    print(y_test)
    print(y_pred)
    print('f1score:%2.f%%'%(f1score*100))

    # save_path  = './components/'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    pickle.dump(model,open(cfg.save_path,'wb'))
    # pickle.dump(zscore_scaler, open('./zscore_scaler'+str(f1score*100)+'.pkl','wb'))
