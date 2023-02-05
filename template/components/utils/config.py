from yacs.config import CfgNode as CN 


__C = CN()
cfg = __C
# 以下两个路径要修改成自己的训练数据的路径，一定要修改，或者将数据映射到以下路径中
# __C.training_ok_path = './components/train/OK'
# __C.training_ng_path = './components/train/NG'
__C.training_ok_path = '../train/OK'
__C.training_ng_path = '../train/NG'
__C.testing_path = r'./components/data'

# 以下路径可以不修改
__C.training_ok_path_mod = './components/training_mod/OK'
__C.training_ng_path_mod = './components/training_mod/NG'
__C.train_feature_path = './components/training_mod/train_feature'
__C.testing_path_mod = './components/training_mod/testing'
__C.test_feature_path = './components/training_mod/test_feature'
__C.train_label_path = './components/training_mod/train_label.csv'
__C.feature_label_path = './components/training_mod/feature_order.csv'
__C.fs_record_path = './components/fs_recode_path'
__C.feature_list_path = './components/feature_list.json'
__C.save_path  = './components/xgb.pkl'