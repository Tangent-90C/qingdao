from utils.config import cfg
from utils.get_test_data import get_test_dataset
import pickle


def inference():
    # 得到训练样本
    print('testing')
    samples = get_test_dataset()
    ids = samples.index
    with open(cfg.save_path, 'rb') as f:
        xgb = pickle.load(f)
        res = xgb.predict(samples)

    return ids, res




if __name__ == '__main__':
    inference()