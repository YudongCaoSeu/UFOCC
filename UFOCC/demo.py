from src.algorithms.ufocc_algo import UFOCC 
from scipy.io import loadmat
import pandas as pd
import numpy as np
import csv
import codecs
from src.utils_general import data_standardize, meta_process_scores, plt_res, minmax_norm
model_configs = {'sequence_length':50, 'stride': 1, 'num_epochs':500, 'batch_size':32, 'lr':1e-4,'alpha':0.01, 'neg_batch_ratio':0.5,'laa':True, 'scc':True}
model = UFOCC(**model_configs)


#//
data = loadmat(r"C:\Users\caoyu\Desktop\一类分类\IMS实验\Main\1-1.mat")
data = data['horiz_signals']
train_data=data[:280,:]
test_data=data[280:,:]
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)
# train_df, test_df = data_standardize(train_df, test_df, remove=False)
# train_df, test_df = train_df.interpolate(), test_df.interpolate()
# train_df, test_df = train_df.bfill(), test_df.bfill()

model.fit(train_df)
score_dic_train = model.predict(train_df)
score_train = score_dic_train['score_t']
score_dic_test = model.predict(test_df)
score_test = score_dic_test['score_t']

def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  # 追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
    print("保存文件成功，处理结束")

score_train=np.reshape(score_train,(len(score_train),1))
data_write_csv(".\\score_train.csv", score_train)
score_test=np.reshape(score_test,(len(score_test),1))
data_write_csv(".\\score_test.csv", score_test)

# data = loadmat(r"C:\Users\caoyu\Desktop\一类分类\IMS实验\Main\F.mat")
# data = data['F']
# train_data=data[:200,:]
# test_data=data[200:,:]
# train_df = pd.DataFrame(train_data)
# test_df = pd.DataFrame(test_data)
# model.fit(train_df)
# score_dic_train = model.predict(train_df)
# score_train = score_dic_train['score_t']
# score_dic_test = model.predict(test_df)
# score_test = score_dic_test['score_t']

# def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
#     file_csv = codecs.open(file_name, 'w+', 'utf-8')  # 追加
#     writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
#     for data in datas:
#         writer.writerow(data)
#     print("保存文件成功，处理结束")

# score_train=np.reshape(score_train,(len(score_train),1))
# data_write_csv(".\\score_train.csv", score_train)
# score_test=np.reshape(score_test,(len(score_test),1))
# data_write_csv(".\\score_test.csv", score_test)