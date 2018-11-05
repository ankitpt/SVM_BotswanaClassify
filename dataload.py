import scipy.io as scio
from pylab import *

data=scio.loadmat('Botswana.mat')
data=data['Botswana']
g_truth = scio.loadmat('Botswana_gt.mat')
g_truth=g_truth['Botswana_gt']

cls, count = np.unique(g_truth, return_counts=True)
cls_no_samp=dict(zip(cls, count))

cls_name=dict()
cls_name={1:'Water',2:'Hippo grass',3:'Floodplain grasses1',4:' Floodplain grasses2',5:'Reeds1',
          6:'Riparian',7:'Firescar2',8:'Island interior',9:'Acacia woodlands',10:'Acacia shrublands',
          11:'Acacia grasslands',12:'Short mopane',13:'Mixed mopane',14:'Exposed soils'}

