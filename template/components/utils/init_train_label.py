import os

import pandas as pd
from utils.config import cfg


def get_file_list(ori_path, label):
    # 由于部分样本sensor检测结果的缺失，使用第一个station得到的sample的列表是全的
    sample_list = os.listdir(os.path.join(ori_path, 'P1000', 'Report_P1000_Time'))
    sample_list = pd.DataFrame({'name': sample_list, 'y': [label] * len(sample_list)})
    return sample_list


def merge_label():
    OK = get_file_list(cfg.training_ok_path, 0)
    NG = get_file_list(cfg.training_ng_path, 1)
    label = pd.concat([OK, NG])
    label.to_csv(cfg.train_label_path, index=False)
