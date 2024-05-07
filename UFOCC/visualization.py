from src.algorithms.ufocc_algo import UFOCC 
from scipy.io import loadmat
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
import csv
import codecs
from src.utils_general import data_standardize, meta_process_scores, plt_res, minmax_norm


model_configs = {'sequence_length':20, 'stride': 1, 'num_epochs':200, 'batch_size':1, 'lr':1e-4,'alpha':0.005, 'neg_batch_ratio':0.5,'laa':True, 'scc':True}
model = UFOCC(**model_configs)

data = loadmat(r"C:\Users\caoyu\Desktop\一类分类\IMS实验\Main\1-3.mat")
data = data['horiz_signals']
train_data=data[:237,:]
test_data=data[237:,:]
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)
# train_df, test_df = data_standardize(train_df, test_df, remove=False)
# train_df, test_df = train_df.interpolate(), test_df.interpolate()
# train_df, test_df = train_df.bfill(), test_df.bfill()

model.fit(train_df)
score_dic_train = model.predict(train_df)
feature_train = score_dic_train['feature'].cpu()
# feature_train=np.reshape(feature_train,feature_train.size(0),feature_train.size(1))
score_dic_test = model.predict(test_df)
feature_test = score_dic_test['feature'].cpu()
# feature_test=np.reshape(feature_test,feature_test.size(0),feature_test.size(1))

feature=np.concatenate([feature_train,feature_test],axis=0)
tsne=TSNE(perplexity=40,n_components=3,n_iter=5000,init='pca')
embs=tsne.fit_transform(feature)

np.save('embs.npy',embs)

embs_1=embs[:1436,:]
embs_2=embs[1436:,:]

fig = plt.figure()

ax=fig.add_subplot(111, projection='3d')
for label in (ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
    label.set_fontname('Times New Roman')

ax.scatter(embs_1[:,0],embs_1[:,1],embs_1[:,2],s=4,label='Representations of Normal period')
ax.scatter(embs_2[:,0],embs_2[:,1],embs_2[:,2],s=4,label='Representations of fault period')
ax.set_xlabel("PC1",fontproperties = 'Times New Roman') 
ax.set_ylabel("PC2",fontproperties = 'Times New Roman') 
ax.set_zlabel("PC3",fontproperties = 'Times New Roman') 
ax.legend(prop={"family": "Times New Roman"})
ax.set_title('Visualization analysis of Bearing 1-3',fontproperties = 'Times New Roman')

xminorLocator = MultipleLocator(5)
ax.xaxis.set_minor_locator(xminorLocator)
yminorLocator = MultipleLocator(5)
ax.yaxis.set_minor_locator(yminorLocator)
plt.show()
fig.savefig('1-3.svg', bbox_inches='tight', pad_inches=0.02,transparent=True)
