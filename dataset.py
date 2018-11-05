import scipy.io as scio
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  
from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.metrics.classification import cohen_kappa_score


class Dataset:
    
    def __init__(self,name,form):
        
        self.name=name
        self.form=form
        
    def load(self):
        
        self.data_dict=scio.loadmat(self.name+self.form)
        self.data=self.data_dict[self.name]
       
def main():
    pass

if __name__ == "__main__":
    bots=Dataset('Botswana','.mat')
    bots.load()

    data=bots.data
    shp=np.shape(data)
    row=shp[0]
    col=shp[1]
    bands=shp[2]
    
    X=data.reshape(row*col,bands)
    
    bots_gt=Dataset('Botswana_gt','.mat')
    bots_gt.load()
    y=bots_gt.data.reshape(row*col,1)
    ind=np.where(y[:,0]!=0)
    X=X[ind]
    y = y[y!=0]
    #Linear SVM with all bands and cv=5
    
    svm=SVC(kernel='linear')
    scores = cross_val_score(svm, X, y, cv=5)
    
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    predictions = cross_val_predict(svm, X, y, cv=5)
    kappa_score = cohen_kappa_score(y,predictions)
    print("Kappa coefficient: %0.2f" % kappa_score)
    #print(confusion_matrix(y_test,y_pred))
    print(classification_report(y, predictions))
    # stuff only to run when not called via 'import' here
    main()



