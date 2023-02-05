import os
import sys
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm
from tsfresh import extract_features
#from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.feature_extraction import MinimalFCParameters as EfficientFCParameters

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


def my_extra(ori_path, tar_path, fill_value=-1000000, maxThreads=None, single=False):
    try:
        station_list = os.listdir(ori_path)
    except:
        print('Please change the paths of training_ok_path and training_ng_path in components/utils/config.py.')
        sys.exit(1)
    if not os.path.exists(tar_path):
        os.makedirs(tar_path)

    # 由于部分样本sensor检测结果的缺失，使用第一个station得到的sample的列表是全的
    sample_list = os.listdir(os.path.join(ori_path, 'P1000', 'Report_P1000_Time'))
    p = Pool(maxThreads)
    pbar = tqdm(total=401, desc='重新生成的训练样本')
    update = lambda *args: pbar.update()

    for station in station_list:
        # print(station)
        for sensor in os.listdir(os.path.join(ori_path, station)):
            if single:
                my_extra_unit(ori_path, tar_path, sample_list, fill_value, station, sensor)
                update()
            else:
                p.apply_async(my_extra_unit, args=(ori_path, tar_path, sample_list, fill_value, station, sensor),
                              callback=update)

    p.close()
    p.join()


def merge_data(maxThreads=None):
    training_ok_path = cfg.training_ok_path
    training_ng_path = cfg.training_ng_path
    training_ok_path_mod = cfg.training_ok_path_mod
    training_ng_path_mod = cfg.training_ng_path_mod

    my_extra(training_ok_path, training_ok_path_mod, maxThreads=maxThreads)
    my_extra(training_ng_path, training_ng_path_mod, maxThreads=maxThreads)


def feature_extraction(maxThreads_for_merge=None, n_jobs_for_extra=2):
    merge_data(maxThreads_for_merge)

    training_ok_path_mod = cfg.training_ok_path_mod
    training_ng_path_mod = cfg.training_ng_path_mod
    train_feature_path = cfg.train_feature_path
    if not os.path.exists(train_feature_path):
        os.makedirs(train_feature_path)

    station_list = os.listdir(training_ok_path_mod)

    setting = EfficientFCParameters()
    setting = filter_settings(setting, level=3)

    pbar = tqdm(total=401,desc='正在抽取特征')

    for station in station_list:
        # print(station)
        for sensor in os.listdir(os.path.join(training_ok_path_mod, station)):
            pbar.desc = f'正在抽取{station}-{sensor}'
            sensor_feature_save_path = os.path.join(train_feature_path, station, sensor)
            kind_to_fc_parameters_save_path = os.path.join(sensor_feature_save_path, f'{station}-{sensor}_features.json')
            tar_path_save_path = os.path.join(sensor_feature_save_path, f'{station}-{sensor}_features.parquet')
            if not os.path.exists(tar_path_save_path):
                if not os.path.exists(sensor_feature_save_path):
                    os.makedirs(sensor_feature_save_path)

                OK_path = os.path.join(training_ok_path_mod, station, sensor, f'{station}-{sensor}.parquet.gzip')
                OK_data = pd.read_parquet(OK_path)

                NG_path = os.path.join(training_ng_path_mod, station, sensor, f'{station}-{sensor}.parquet.gzip')
                NG_data = pd.read_parquet(NG_path)

                datas = pd.concat([OK_data, NG_data])

                features = extract_features(datas, default_fc_parameters=setting, impute_function=impute, column_id='name',
                                            column_sort='Time', n_jobs=n_jobs_for_extra)

                kind_to_fc_parameters = tsfresh.feature_extraction.settings.from_columns(features)
                info_json = json.dumps(kind_to_fc_parameters, sort_keys=False, indent=4, separators=(',', ': '))
                with open(kind_to_fc_parameters_save_path, 'w') as f:
                    f.write(info_json)
                features.to_parquet(tar_path_save_path)
            pbar.update()
