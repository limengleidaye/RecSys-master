import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

df=pd.read_csv('/MF/res/log_noise.csv', engine='python')
df_non_noise=pd.read_csv('/MF/res/log_without_noise.csv', engine='python')

loss=df['val_loss']
loss_non_noise=df_non_noise['val_loss']

plt.figure(figsize=(20,8),dpi=80)

_ylabels=np.sqrt(loss)
_xlabels=range(len(loss))

_ylabels_non_noise=np.sqrt(loss_non_noise)
_xlabels_non_noise=range(len(loss_non_noise))

plt.ylabel('rmse')
plt.xlabel('epoch')

plt.xticks(range(len(loss)))

plt.plot(_xlabels,_ylabels,label="加噪声")
plt.plot(_xlabels_non_noise,_ylabels_non_noise,linestyle='--',label="不加噪声")

plt.legend()
plt.savefig('./compare in noise.png')
plt.show()
