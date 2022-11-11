#!/usr/bin/env python
# coding: utf-8

# ## 1.利物浦大学---离子交换
# ![img](header.png)

# > 背景:通过每个时间点的电信号来预测离子通道的个数。通过电流来观察离子通道的个数，这些个数与一些疾病有关，比如说癌症。

# ## 2.导入库

# In[1]:


import tensorflow as tf
from tensorflow.keras.layers import *
import pandas as pd
import numpy as np
import random
from tensorflow.keras.callbacks import Callback, LearningRateScheduler,ModelCheckpoint
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras import losses, models, optimizers

import tensorflow_addons as tfa
import gc
import os
# 设置CUDA属性
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

## 一些设置
import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)

### 打印一些文件夹和文件
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## 3.一些配置

# In[2]:


# configurations and main hyperparammeters
EPOCHS = 120
NNBATCHSIZE = 16
GROUP_BATCH_SIZE = 4000
SEED = 321
LR = 0.0015
SPLITS = 5

# 取种子，方便代码复现
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)


# ## 4.数据读取:3个部分

# In[ ]:


# 读取数据
def read_data():
    # 读取经过处理后的干净数据(除去drift的数据)
    train = pd.read_csv('../input/data-without-drift/train_clean.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})
    test  = pd.read_csv('../input/data-without-drift/test_clean.csv', dtype={'time': np.float32, 'signal': np.float32})
    sub  = pd.read_csv('../input/liverpool-ion-switching/sample_submission.csv', dtype={'time': np.float32})
    # 读取随机森林的输出概率结果
    Y_train_proba = np.load("../input/ion-shifted-rfc-proba/Y_train_proba.npy")
    Y_test_proba = np.load("../input/ion-shifted-rfc-proba/Y_test_proba.npy")
    # 把这个结果添加到train和test里面
    for i in range(11):
        train[f"proba_{i}"] = Y_train_proba[:, i]
        test[f"proba_{i}"] = Y_test_proba[:, i]

    return train, test, sub


# ## 5.一些数据处理和特征工程

# In[ ]:



# 以每4000行为一个batch
def batching(df, batch_size):
    df['group'] = df.groupby(df.index//batch_size, sort=False)['signal'].agg(['ngroup']).values
    df['group'] = df['group'].astype(np.uint16)
    return df

# 神经网络需要正规化，剪均值去方差
def normalize(train, test):
    train_input_mean = train.signal.mean()
    train_input_sigma = train.signal.std()
    train['signal'] = (train.signal - train_input_mean) / train_input_sigma
    test['signal'] = (test.signal - train_input_mean) / train_input_sigma
    return train, test

# 得到lead和lag数据，lead是前几个点的数据，lag是后几个点的数据
def lag_with_pct_change(df, windows):
    for window in windows:    
        df['signal_shift_pos_' + str(window)] = df.groupby('group')['signal'].shift(window).fillna(0)
        df['signal_shift_neg_' + str(window)] = df.groupby('group')['signal'].shift(-1 * window).fillna(0)
    return df

# main module to run feature engineering. Here you may want to try and add other features and check if your score imporves :).
def run_feat_engineering(df, batch_size):
    # 创建batch
    df = batching(df, batch_size = batch_size)
    # 得到lead和lag数据
    df = lag_with_pct_change(df, [1, 2, 3])
    # 得到平方数据，这是种能量的表现
    df['signal_2'] = df['signal'] ** 2
    return df

# 除了'index', 'group', 'open_channels', 'time'其他列进行选择，同时使用均值进行空值填充
def feature_selection(train, test):
    features = [col for col in train.columns if col not in ['index', 'group', 'open_channels', 'time']]
    train = train.replace([np.inf, -np.inf], np.nan)
    test = test.replace([np.inf, -np.inf], np.nan)
    print(len(features))
    for feature in features:
        feature_mean = pd.concat([train[feature], test[feature]], axis = 0).mean()
        train[feature] = train[feature].fillna(feature_mean)
        test[feature] = test[feature].fillna(feature_mean)
    return train, test, features


# ## 6.wavenet的网络介绍

# Wavenet模型是一种序列生成模型，可以用于语音生成建模。在语音合成的声学模型建模中，Wavenet可以直接学习到采样值序列的映射，因此具有很好的合成效果。目前wavenet在语音合成声学模型建模，vocoder方面都有应用，在语音合成领域有很大的潜力。



# Wavenet模型主要成分是这种卷积网络，每个卷积层都对前一层进行卷积，卷积核越大，层数越多，时域上的感知能力越强，感知范围越大。在生成过程中，每生成一个点，把该点放到输入层最后一个点继续迭代生成即可。


# ### 6.2 激活函数

# ![img](activation.png)

# ### 6.3.wavenet block

# ![img](wavenet_block.png)

# ## 7.wavenet代码

# In[ ]:



## 神经网络
def Classifier(shape_):
    # cbr结构 Conv1D + BatchNormalization + Relu
    def cbr(x, out_layer, kernel, stride, dilation):
        x = Conv1D(out_layer, kernel_size=kernel, dilation_rate=dilation, strides=stride, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x
    
    ## wavenet 模块
    def wave_block(x, filters, kernel_size, n):
        # 膨胀因子
        dilation_rates = [2**i for i in range(n)]
        x = Conv1D(filters = filters,
                   kernel_size = 1,
                   padding = 'same')(x)
        res_x = x
        for dilation_rate in dilation_rates:
            tanh_out = Conv1D(filters = filters,
                              kernel_size = kernel_size,
                              padding = 'same', 
                              activation = 'tanh', 
                              dilation_rate = dilation_rate)(x)
            sigm_out = Conv1D(filters = filters,
                              kernel_size = kernel_size,
                              padding = 'same',
                              activation = 'sigmoid', 
                              dilation_rate = dilation_rate)(x)
            x = Multiply()([tanh_out, sigm_out])
            x = Conv1D(filters = filters,
                       kernel_size = 1,
                       padding = 'same')(x)
            res_x = Add()([res_x, x])
        return res_x
    
    ###从这里开始搭建主体网络
    inp = Input(shape = (shape_))
    # 第一个cbr
    x1 = cbr(inp, 64,3, 1, 2)
    x1 = BatchNormalization()(x1)
    # 第二个cbr
    x2 = cbr(inp, 64,5, 1, 2)
    x2 = BatchNormalization()(x2)
    # 将两个cbr相连接
    x = Concatenate()([x1,x2])
    # 第一个wavenet block
    x = wave_block(x, 24, 3, 12)
    x = BatchNormalization()(x)
     # 第二个wavenet block
    x = wave_block(x, 32, 3, 8)
    x = BatchNormalization()(x)
     # 第三个wavenet block
    x = wave_block(x, 64, 3, 4)
    x = BatchNormalization()(x)
     # 第四个wavenet block
    x = wave_block(x, 128, 3, 1)
    # 添加一个cbr
    x = cbr(x, 32, 3, 1, 1)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    # 最后只有11个，因为总共只有11类
    out = Dense(11, activation = 'softmax', name = 'out')(x)
    
    model = models.Model(inputs = inp, outputs = out)
    # adam作为优化器
    opt = Adam(lr = LR)
    # SWA进行优化器优化
    opt = tfa.optimizers.SWA(opt)
    # 模型编译
    model.compile(loss = losses.CategoricalCrossentropy(), optimizer = opt, metrics = ['accuracy'])
#     model.compile(loss = categorical_focal_loss(11),optimizer = opt, metrics = ['accuracy'])
    return model


# ## 8.学习率以及回调函数

# In[ ]:



# f学习率schedule，一种时间表
def lr_schedule(epoch):
    if epoch < 30:
        lr = LR
    elif epoch < 40:
        lr = LR / 3
    elif epoch < 50:
        lr = LR / 5
    elif epoch < 60:
        lr = LR / 7
    elif epoch < 70:
        lr = LR / 9
    elif epoch < 80:
        lr = LR / 11
    elif epoch < 90:
        lr = LR / 13
    else:
        lr = LR / 100
    return lr

# 定义一个keras里面可以显示MicroF1的回调函数
class MacroF1(Callback):
    def __init__(self, model, inputs, targets):
        self.model = model
        self.inputs = inputs
        self.targets = np.argmax(targets, axis = 2).reshape(-1)
        
    def on_epoch_end(self, epoch, logs):
        pred = np.argmax(self.model.predict(self.inputs), axis = 2).reshape(-1)
        score = f1_score(self.targets, pred, average = 'macro')
        print(f'F1 Macro Score: {score:.5f}')


# In[3]:




# main function to perfrom groupkfold cross validation (we have 1000 vectores of 4000 rows and 8 features (columns)). Going to make 5 groups with this subgroups.
# 使用groupkfold做交叉验证，
def run_cv_model_by_batch(train, test, splits, batch_col, feats, sample_submission, nn_epochs, nn_batch_size):
    # 设置种子
    seed_everything(SEED)
    # 使用比较紧凑的图，可以节省gpu显存。默认tensorflow会使用整个图，比较占gpu
    K.clear_session()
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=2,inter_op_parallelism_threads=2)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)
    tf.compat.v1.keras.backend.set_session(sess)
    
    # 一些设置
    oof_ = np.zeros((len(train), 11)) 
    preds_ = np.zeros((len(test), 11))
    target = ['open_channels']
    # 根据group进行kfold
    group = train['group']
    kf = GroupKFold(n_splits=5)
    splits = [x for x in kf.split(train, train[target], group)]
    
    ## 把split的结果保存到new_splits里面
    new_splits = []
    for sp in splits:
        new_split = []
        new_split.append(np.unique(group[sp[0]]))
        new_split.append(np.unique(group[sp[1]]))
        new_split.append(sp[1])    
        new_splits.append(new_split)
        
    # 把open_channels变成one-hot编码
    tr = pd.concat([pd.get_dummies(train.open_channels), train[['group']]], axis=1)

    tr.columns = ['target_'+str(i) for i in range(11)] + ['group']
    target_cols = ['target_'+str(i) for i in range(11)]
    train_tr = np.array(list(tr.groupby('group').apply(lambda x: x[target_cols].values))).astype(np.float32)
    train = np.array(list(train.groupby('group').apply(lambda x: x[feats].values)))
    test = np.array(list(test.groupby('group').apply(lambda x: x[feats].values)))

    for n_fold, (tr_idx, val_idx, val_orig_idx) in enumerate(new_splits[0:], start=0):
        train_x, train_y = train[tr_idx], train_tr[tr_idx]
        valid_x, valid_y = train[val_idx], train_tr[val_idx]
        # 清理一下内存，垃圾回收
        gc.collect()
        # shape的大小，为输入数据的维度
        shape_ = (None, train_x.shape[2]) 
        model = Classifier(shape_)
        
        # 使用checkpoint保存最好的模型结果
        checkpoint = ModelCheckpoint(filepath='wavenet.h5',monitor='val_accuracy',mode='auto' ,save_best_only='True')
        # 使用自定义的schedule计划
        cb_lr_schedule = LearningRateScheduler(lr_schedule)
        # 模型拟合
        model.fit(train_x,train_y,
                  epochs = nn_epochs,
                  callbacks = [cb_lr_schedule,MacroF1(model, valid_x, valid_y),checkpoint], # adding custom evaluation metric for each epoch
                  batch_size = nn_batch_size,verbose = 2,
                  validation_data = (valid_x,valid_y))
        # 加载最好的模型
        model.load_weights("wavenet.h5")
        preds_f = model.predict(valid_x)
        # 计算F1分数，并且打印
        f1_score_ = f1_score(np.argmax(valid_y, axis=2).reshape(-1),  np.argmax(preds_f, axis=2).reshape(-1), average = 'macro') # need to get the class with the biggest probability
        print(f'Training fold {n_fold + 1} completed. macro f1 score : {f1_score_ :1.5f}')
        preds_f = preds_f.reshape(-1, preds_f.shape[-1])
        oof_[val_orig_idx,:] += preds_f
        te_preds = model.predict(test)
        te_preds = te_preds.reshape(-1, te_preds.shape[-1])           
        preds_ += te_preds / SPLITS
        
    
    f1_score_ = f1_score(np.argmax(train_tr, axis = 2).reshape(-1),  np.argmax(oof_, axis = 1), average = 'macro') # axis 2 for the 3 Dimension array and axis 1 for the 2 Domension Array (extracting the best class)
    print(f'Training completed. oof macro f1 score : {f1_score_:1.5f}')
    # 把测试集预测结果保存到sample_submission里面
    sample_submission['open_channels'] = np.argmax(preds_, axis = 1).astype(int)
    sample_submission.to_csv('submission_wavenet1.csv', index=False, float_format='%.4f')
    
# 运行整个项目的函数
def run_everything():
    
    print('Reading Data Started...')
    train, test, sample_submission = read_data()
    train, test = normalize(train, test)
    print('Reading and Normalizing Data Completed')
        
    print('Creating Features')
    print('Feature Engineering Started...')
    train = run_feat_engineering(train, batch_size = GROUP_BATCH_SIZE)
    test = run_feat_engineering(test, batch_size = GROUP_BATCH_SIZE)
    train, test, features = feature_selection(train, test)
    print('Feature Engineering Completed...')
        
   
    print(f'Training Wavenet model with {SPLITS} folds of GroupKFold Started...')
    run_cv_model_by_batch(train, test, SPLITS, 'group', features, sample_submission, EPOCHS, NNBATCHSIZE)
    print('Training completed...')
        
run_everything()


# In[ ]:





# In[ ]:




