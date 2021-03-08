import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

df=pd.read_csv('F:/OfficialApps/workspace/pythonProject/RecSys-master/MF/res/log_SGD.csv',engine='python')
df_non_noise=pd.read_csv('F:/OfficialApps/workspace/pythonProject/RecSys-master/MF/res/log_non_noise.csv',engine='python')

loss=df['val_loss']
loss_non_noise=df_non_noise['val_loss']

plt.figure(figsize=(20,8),dpi=80)

_ylabels=np.sqrt(loss)
_xlabels=range(len(loss))

_ylabels_non_noise=np.sqrt(loss_non_noise)
_xlabels_non_noise=range(len(loss_non_noise))

plt.ylabel('rmse')
plt.xlabel('epoch')

plt.xticks(_xlabels[::5],range(len(loss))[::5])

plt.plot(_xlabels,_ylabels,label="SGD")
plt.plot(_xlabels_non_noise,_ylabels_non_noise,linestyle='--',label="Adam")

plt.legend()
plt.savefig('./compare in opt.png')
plt.show()
