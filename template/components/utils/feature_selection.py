from utils.config import cfg
import os
import numpy as np
import json 


def var_(samples,mean_):
    return np.mean((samples-mean_)*(samples-mean_),axis=0)


def processing_record(path1,path2):
    save_base_path = cfg.fs_record_path
    for station in os.listdir(path1):
        if not os.path.exists(os.path.join(save_base_path,station)):
            os.makedirs(os.path.join(save_base_path,station))
        for sensor in os.listdir(os.path.join(path1,station)):
            sensor_recode_path = os.path.join(save_base_path,station,sensor+'.txt')
            if not os.path.exists(sensor_recode_path):
                print(station,sensor)
                samples_ok = []
                for sample_name in os.listdir(os.path.join(path1,station,sensor)):
                    sample_path = os.path.join(path1,station,sensor,sample_name)
                    if os.path.exists(sample_path):
                        if np.count_nonzero(np.isnan(np.load(sample_path)))==0:
                            sample_ = np.load(sample_path)
                            if len(sample_)==0:
                                continue
                            else:
                                samples_ok.append(list(sample_))
                samples_mean_ok = np.mean(samples_ok,axis=0)
                # samples_var_mean1 = var_(samples_ok,samples_mean_ok).mean()
                samples_ng = []
                for sample_name in os.listdir(os.path.join(path2,station,sensor)):
                    sample_path = os.path.join(path2,station,sensor,sample_name)
                    if os.path.exists(sample_path):
                        if np.count_nonzero(np.isnan(np.load(sample_path)))==0:
                            sample_ = np.load(sample_path)
                            if len(sample_)==0:
                                continue
                            else:
                                samples_ng.append(list(sample_))

                
                if np.var(samples_ng)==0:
                    res=[0]
                elif np.var(samples_ok)==0:
                    res = [1.0]
                else:
                    res = [np.var(samples_ng)/np.var(samples_ok)]
                print(np.var(samples_ok))
                print(np.var(samples_ng))
                print(res)
                np.savetxt(sensor_recode_path,res)
                sensor_mean_recode_path = os.path.join(save_base_path,station,sensor+'_mean.txt')
                np.savetxt(sensor_mean_recode_path,[samples_mean_ok])
                sensor_mean_recode_path = os.path.join(save_base_path,station,sensor+'_var.txt')
                np.savetxt(sensor_mean_recode_path,[np.var(samples_ok,axis=0)])

            
def processing_select():
    record_base_path = cfg.fs_record_path
    th_list = []
    for station in os.listdir(cfg.training_ok_path_mod):
        for sensor in os.listdir(os.path.join(cfg.training_ok_path_mod,station)):
            file_path = os.path.join(record_base_path,station,sensor+'.txt')
            th_list.append(np.loadtxt(file_path))
    th_list.sort(reverse=True)
    th = th_list[50]
    feature_list = {}
    for station in os.listdir(cfg.training_ok_path_mod):
        feature_list_sub = []
        for sensor in os.listdir(os.path.join(cfg.training_ok_path_mod,station)):
            file_path = os.path.join(record_base_path,station,sensor+'.txt')
            if np.loadtxt(file_path)>th:
                feature_list_sub.append(sensor)
        feature_list[station]=feature_list_sub
    print(feature_list)
    with open(cfg.feature_list_path,'w',encoding='utf-8') as f:
        json.dump(feature_list,f)
                
                
                


    

def feature_selection():
    training_ok_path_mod = cfg.training_ok_path_mod
    training_ng_path_mod = cfg.training_ng_path_mod
    # 得到正样本所有的均值和最大方差
    processing_record(training_ok_path_mod,training_ng_path_mod)
    # sample_list = os.listdir(os.path.join(ori_path,'P1000','Report_P1000_Time'))
    # station_list = os.listdir(ori_path)
    # 根据正样本的均值，得到负样本的方差，并统计超过最大方差的次数
    # 根据统计次数，找出前50个特征
    processing_select()
    
    return 0