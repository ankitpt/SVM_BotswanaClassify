import time
from dataset import Dataset
from scipy.stats import norm
import numpy as np1
import pandas as pd
import numpy as np
from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.metrics.classification import cohen_kappa_score

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
df2=np.array(df2)

start_time = time.time()
svm=SVC(kernel='linear')
scores = cross_val_score(svm, df2, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predictions = cross_val_predict(svm, df2, y, cv=5)
kappa_score = cohen_kappa_score(y,predictions)
print("Kappa coefficient: %0.2f" % kappa_score)
print(classification_report(y, predictions))

print("--- %s seconds ---" % (time.time() - start_time))