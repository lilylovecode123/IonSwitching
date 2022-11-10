#!/usr/bin/env python
# coding: utf-8

# ### 模型融合
# 使用投票的机制，进行模型融合。首先多跑几次wavenet网络的结果，然后对这些结果进行投票

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
### 导入库函数
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# visualize
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
from matplotlib import pyplot
from matplotlib.ticker import ScalarFormatter
sns.set_context("talk")
style.use('fivethirtyeight')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from scipy import stats
from tqdm import tqdm_notebook as tqdm
import os

# Any results you write to the current directory are saved as output.


# # 加载模型的文件

# In[2]:


# load submission files
submit = pd.read_csv("../input/liverpool-ion-switching/sample_submission.csv")

# high scoring public kernels
### 几个需要融合的模型
paths = {
    "wavenet1": "submission_118.csv",
    "wavenet2": "submission_136.csv",
    "wavenet3": "submission_127.csv",
    "wavenet4": "submission_121.csv",
    "wavenet5": "test.csv",
}
weights = np.array([1,1.5,1.2,1.0,3.3]) # 单个模型效果越好，分数越高
subs = submit.copy()
for i, p in enumerate(paths.keys()):
    tmp = pd.read_csv(paths[p])
    print(p)
    subs[f"{p}"] = tmp["open_channels"]

subs.drop(columns=["time", "open_channels"], inplace=True)


# # 每个文件的相关度分析

# In[3]:


# Compute the correlation matrix
corr = subs.corr()

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(15, 12))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, cmap=cmap, annot=True, fmt="g",
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
ax.set_ylim(corr.shape[0], 0)
plt.yticks(rotation=0)


# # 投票出新结果

# In[4]:


# pandas weighted voting implementation
def weighted_voting(row):
    h = np.histogram(row.values, weights=weights)
    return np.round(h[1][np.argmax(h[0])])


# In[5]:


submit["open_channels"] = subs.apply(weighted_voting, axis=1)


# # 保存提交结果

# In[6]:


submit["open_channels"] = submit["open_channels"].astype(int)

submit.to_csv('submission.csv', index=False, float_format='%.4f')


# In[1]:


submit["open_channels"].hist(alpha=0.5) ### 查看一下结果的分布


# In[ ]:




