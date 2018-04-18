# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 21:58:29 2018

@author: brian
"""

def gradientComputer_1(hList, gList, xList, zpList, dEdyhat):    
    #L1 =  L2   = 0
    penalty = 0
    m = len(hList)
    s = dim(hList[m - 1])[1]
    for r in range(m-1, -1, -1):
        
        shape = dim(hList[r])
        if r > 0:
            A = augment(xList[r], 1)
        else:
            A = xList[r]
        zeros = np.matrix(np.zeros( dim(zpList[r]) ))   
        
        #E = np.matrix(np.zeros( shape))       
        for i in range(shape[0]):
            for j in range(shape[1]):
                #dyhatdh = np.matrix(np.zeros(dim(zpList[r])))   
                dyhatdh = zeros.copy()   
                dyhatdh[:,j] = np.multiply( A[:,i], zpList[r][:,j])
                #E[i,j] = 1
            
                #dyhatdh = np.multiply( A*E, zpList[r])
                
                for k in range(r+1,m,1):
                    
                    dyhatdh = np.multiply(dyhatdh * hList[k][1:,:], zpList[k])
                gList[r][i,j] =  np.sum([dyhatdh[:,k].T*dEdyhat[:,k] for k in range(s)]) 
                
                #E[i,j] = 0
    return gList