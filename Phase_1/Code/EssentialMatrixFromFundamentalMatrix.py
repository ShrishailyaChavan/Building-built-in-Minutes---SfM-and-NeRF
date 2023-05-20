import numpy as np

def getEssentialMatrix(K, F):
    E = np.dot(K.T, np.dot(F, K))
    U,s,V = np.linalg.svd(E)
    s = [1,1,0]
    E_corrected = np.dot(U,np.dot(np.diag(s),V))
    return E_corrected