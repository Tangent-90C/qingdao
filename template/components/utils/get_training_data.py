import json
import os

import pandas as pd
from utils.config import cfg


def get_training_data(mid_path):
    with open(cfg.feature_list_path, 'r', encoding='utf8') as f:
        feature_list = json.load(f)

    labels = pd.read_csv(cfg.train_label_path)
    labels.set_index('name', inplace=True, drop=True)
    station_list = feature_list.keys()

    samples = []
    for station in station_list:
        # Pxxxx
        for sensor in feature_list[station]:
            file_path = os.path.join(cfg.train_feature_path, station, sensor, "{}-{}_features.parquet".format(station, sensor))
            samples.append(pd.read_parquet(file_path))

    for df in samples:
        labels = labels.join(df)

    pd.Series(labels.drop('y', axis=1).columns).to_csv(cfg.feature_label_path)
    return labels.drop('y', axis=1), labels['y']
