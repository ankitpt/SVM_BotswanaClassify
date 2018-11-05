from dataset import Dataset
from scipy.stats import norm
import numpy as np
import pandas as pd

def get_wts(arr):
    return norm.cdf(arr,np.mean(arr),np.std(arr))

bots=Dataset("Botswana",".mat")
bots.load()
hypd=bots.data


shp=np.shape(hypd)
row=shp[0]
col=shp[1]
bands=shp[2]
    
hypd=hypd.reshape(row*col,bands)
    
bots_gt=Dataset('Botswana_gt','.mat')
bots_gt.load()
y=bots_gt.data.reshape(row*col,1)
ind=np.where(y[:,0]!=0)
X_hypd=hypd[ind]
y = y[y!=0]


df = pd.DataFrame(X_hypd)

df2=pd.DataFrame()

band=dict()
band['Blue']=np.linspace(490,530,5)
band['Green']=np.linspace(530,590,7)
band['Red']=np.linspace(630,680,6)
band['NIR']=np.linspace(850,880,4)



wblue=np.ediff1d(get_wts(band['Blue']))
wgreen=np.ediff1d(get_wts(band['Green']))
wred=np.ediff1d(get_wts(band['Red']))
wnir=np.ediff1d(get_wts(band['NIR']))




df2['Blue'] = df.iloc[:, 0:4].multiply(wblue, axis=1).sum(axis=1)
df2['Green'] = df.iloc[:, 4:10].multiply(wgreen, axis=1).sum(axis=1)
df2['Red'] = df.iloc[:, 14:19].multiply(wred, axis=1).sum(axis=1)
df2['NIR'] = df.iloc[:, 36:39].multiply(wnir, axis=1).sum(axis=1)