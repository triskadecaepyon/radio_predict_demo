import numpy as np
import scipy
from sklearn import svm

def train_simple_model(full_data, mask):
    
    clf = svm.SVC(gamma='scale')
    clf.fit(full_data.T, mask.astype('int'))

    return clf
    
