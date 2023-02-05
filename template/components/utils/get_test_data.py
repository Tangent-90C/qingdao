import os

import pandas as pd
from tqdm import tqdm
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute
import tsfresh
import json
from utils.config import cfg
from utils.filter import filter_settings


def my_extra_unit(ori_path, tar_path, sample_list, fill_value, station, sensor):
    print(station, sensor)
    sensor_save_path = os.path.join(tar_path, station, sensor)
    tar_path_save_path = os.path.join(sensor_save_path, f'{station}-{sensor}.parquet.gzip')
    if os.path.exists(tar_path_save_path):
        return
    if not os.path.exists(sensor_save_path):
        os.makedirs(sensor_save_path)

    datas = []
    for sample_name in tqdm(sample_list, leave=False, delay=3):
        sample_path = os.path.join(ori_path, station, sensor, sample_name)
        if os.path.exists(sample_path):
            sample_onesensor_csv = pd.read_csv(sample_path)
            if len(sample_onesensor_csv) == 0:
                sample_onesensor_csv.iloc[:, 0] = [fill_value] * 2
            sample_onesensor_csv.columns = [f'{station}-{sensor}']
            sample_onesensor_csv['name'] = sample_name
            sample_onesensor_csv['Time'] = range(1, len(sample_onesensor_csv) + 1)
            datas.append(sample_onesensor_csv)
        else:
            sample_onesensor_csv = pd.DataFrame({f'{station}-{sensor}': [fill_value] * 2})
            sample_onesensor_csv['name'] = sample_name
            sample_onesensor_csv['Time'] = range(1, len(sample_onesensor_csv) + 1)
            datas.append(sample_onesensor_csv)

    datas = pd.concat(datas, axis=0)
    datas['name'] = datas['name'].astype('category')
    datas.fillna(inplace=True, method='ffill')
    datas.fillna(inplace=True, method='bfill')
    datas.to_parquet(tar_path_save_path, compression='gzip')


def my_extra(ori_path, tar_path, fill_value=-1000000):
    if not os.path.exists(tar_path):
        os.makedirs(tar_path)

    with open(cfg.feature_list_path, 'r', encoding='utf8') as f:
        feature_list = json.load(f)

    station_list = feature_list.keys()
    # 由于部分样本sensor检测结果的缺失，使用第一个station得到的sample的列表是全的
    sample_list = os.listdir(os.path.join(ori_path, 'P1000', 'Report_P1000_Time'))

    for station in station_list:
        # print(station)
        for sensor in feature_list[station]:
            my_extra_unit(ori_path, tar_path, sample_list, fill_value, station, sensor)



def merge_data(maxThreads=None):
    testing_path = cfg.testing_path
    testing_path_mod = cfg.testing_path_mod

    my_extra(testing_path, testing_path_mod)


def get_test_dataset(maxThreads_for_merge=None, n_jobs_for_extra=1):
    merge_data(maxThreads_for_merge)

    testing_path_mod = cfg.testing_path_mod
    test_feature_path = cfg.test_feature_path
    if not os.path.exists(test_feature_path):
        os.makedirs(test_feature_path)

    setting = EfficientFCParameters()
    setting = filter_settings(setting, level=3)


    with open(cfg.feature_list_path, 'r', encoding='utf8') as f:
        feature_list = json.load(f)

    station_list = feature_list.keys()

    samples = []
    for station in station_list:
        # Pxxxx
        for sensor in feature_list[station]:
            sensor_feature_save_path = os.path.join(cfg.train_feature_path, station, sensor)
            kind_to_fc_parameters_save_path = os.path.join(sensor_feature_save_path, f'{station}-{sensor}_features.json')
            with open(kind_to_fc_parameters_save_path, 'r') as load_f:
                kind_to_fc_parameters = json.load(load_f)

            test_path = os.path.join(testing_path_mod, station, sensor, f'{station}-{sensor}.parquet.gzip')
            test_data = pd.read_parquet(test_path)
            features = extract_features(test_data, default_fc_parameters=setting, impute_function=impute, column_id='name',
                                        column_sort='Time', n_jobs=n_jobs_for_extra,kind_to_fc_parameters=kind_to_fc_parameters)

            samples.append(features)

    data = samples[0]
    for i in samples[1:]:
        data = data.join(i)

    label_order = pd.read_csv(cfg.feature_label_path,index_col=0).iloc[:,0]
    data = data[label_order]

    return data