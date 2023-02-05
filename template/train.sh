# 记得修改 components/utils/feature_extraction.py中训练数据的路径

# components/training_mod 存储了从原始特征中提取的特征，
# 该算法简单提取了每个传感器记录中的最大值最小值和均值，
# 该文件夹内的文件如果存在，则不会执行，
# 如果要重新提取特征，需删除该文件夹；如不需要重新提取特征，可将其注释掉

# rm -rf components/training_mod

# components/fs_recode_path 存储了OK样本的均值、方差和NG样本方差与OK样本方差的比值，
# OK样本的均值、方差用于补全缺失特征， 
# PS：注意，使用均值和方差补全特征的操作会引入变量，导致每次测试结果不同（相同模型情况下）
# NG样本方差与OK样本方差的比值用于选取前50个特征（或者说传感器）
# 同样地，其中的文件如果存在则不会重复执行，
# 如需重新计算，需删掉该文件夹，否则，注释掉

# rm -rf components/fs_recode_path


python components/train.py
