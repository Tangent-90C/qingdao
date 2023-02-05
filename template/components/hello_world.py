import pandas as pd
#import sys
#sys.path.append('components')
from test import inference



# 补充说明：
# 1.提交文件压缩包文件格式为。zip，记得查看下文件解压出来是个template名字的文件夹，里边是具体的文件，包括compponents，build.sh等
# 3.运行run.sh文件后生成的两个文件（*.csv和finish.txt）记得删除，至少要删除一个
# 4.版本依赖记得写全


ids,results = inference()


# ids = [
#     '7ebfef6101d03140b3d07d550857e584.csv',
#     '855e756747da36f98254c7255cd603b7.csv',
#     'fdc30cbecfc533d1b13c222cd0b3508a.csv'
# ]
# results = [0,1,0] 
dataframe = pd.DataFrame({'id':ids,'result':results})
dataframe.to_csv("test_predict.csv",index=False,sep=',')  # 将DataFrame存储为csv,index表示是否显示行名，default=True
