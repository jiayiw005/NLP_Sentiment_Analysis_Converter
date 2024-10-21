import numpy
import pandas as pd
#数据处理库
import re

import jieba

#可视化库
import stylecloud

import matplotlib.pyplot as plt

import seaborn as sns

#matplotlib inline
from pyecharts.charts import *

from pyecharts import options as opts

from pyecharts.globals import ThemeType

from IPython.display import Image

#文本挖掘库

from snownlp import SnowNLP

from gensim import corpora,models




#error_bad_lines参数可忽略异常行

df = pd.read_csv("/Users/nb_wjy/desktop/CTB/双减老师.csv",header=None,sep='|',error_bad_lines=False)

df = df.iloc[:,[0]] #选择用户名和内容列

df = df.drop_duplicates() #删除重复行

df = df.dropna() #删除存在缺失值的行

df.columns = ["content"] #对字段进行命名

df['score'] = df["content"].apply(lambda x:SnowNLP(x).sentiments)





plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

plt.figure(figsize=(12, 6)) #设置画布大小

rate = df['score']

ax = sns.distplot(rate,

            hist_kws={'color':'blue','label':'直方图'},

            kde_kws={'color':'green','label':'密度曲线'},

            bins=200) #参数color样式为salmon，bins参数设定数据片段的数量

ax.set_title('Sentimental Inclination of "shuangjianlaoshi" by CONVERTERs')

plt.show()

print(df['score'])


