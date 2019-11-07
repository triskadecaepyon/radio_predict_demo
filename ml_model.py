import numpy as np
import scipy
from sklearn import svm

def train_simple_model(full_data, mask):
    """
    A placeholder for a simple ML model.  
    Please note this is training time, which
    responds to training size significantly more
    than inference.  

    Other details: with random noise you can get
    a situtation where only one class for the classifier
    might happen--in this case ignore it as the random
    noise will eventually create two classes again.
    """
    
    clf = svm.SVC(gamma='scale')
    clf.fit(full_data.T, mask.astype('int'))

    return clf
    
